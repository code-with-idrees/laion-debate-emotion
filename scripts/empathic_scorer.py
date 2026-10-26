"""Empathic-Insight-Voice-Plus scorer (batched).

55 affect/attribute experts (Empathic-Insight-Voice-Small) + 4 audio-quality
experts (Empathic-Insight-Voice-Plus), all on frozen BUD-E-Whisper embeddings.

The 55 affect/attribute experts share one architecture (1500x768 -> 64 -> 64 ->
32 -> 16 -> 1).  Loading them as 55 separate 74M-parameter modules wastes both
time and VRAM, so their projection weights are fused into one bank and applied
with 54 matmuls per batch.  The tiny MLP heads are applied in the same loop.
"""
from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from transformers import WhisperFeatureExtractor, WhisperModel

sys.path.insert(0, str(Path(__file__).parent))
import config as C

QUALITY_FILES = {
    "Overall_Quality": "model_score_overall_quality_best.pth",
    "Speech_Quality": "model_score_speech_quality_best.pth",
    "Background_Quality": "model_score_background_quality_best.pth",
    "Content_Enjoyment": "model_score_content_enjoyment_best.pth",
}


def pretty_name(stem: str) -> str:
    s = stem.replace("model_", "").replace("_best", "")
    s = s.replace("_vs._", " vs ").replace("_", " ")
    s = s.replace(".", "").strip()
    return s


def load_state(path: Path) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if any(k.startswith("_orig_mod.") for k in sd):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    return sd


class EmpathicScorer:
    def __init__(self, device: str = "cuda", dtype=torch.float16,
                 emotion_dir: Path | None = None, plus_dir: Path | None = None,
                 bude_dir: Path | None = None):
        self.device = device
        self.dtype = dtype
        # Voice-Plus bundles all 59 experts, so one directory serves both roles.
        emotion_dir = Path(emotion_dir or C.EMPATHIC_PLUS_DIR)
        plus_dir = Path(plus_dir or C.EMPATHIC_PLUS_DIR)
        bude_dir = Path(bude_dir or C.BUDE_DIR)

        self.fe = WhisperFeatureExtractor.from_pretrained(str(bude_dir))
        whisper = WhisperModel.from_pretrained(str(bude_dir), low_cpu_mem_usage=True)
        self.encoder = whisper.get_encoder().to(device).eval()
        del whisper
        gc.collect()
        torch.cuda.empty_cache()

        # --- emotion experts ------------------------------------------------
        files = sorted(p for p in emotion_dir.glob("model_*_best.pth")
                       if not p.name.startswith("model_score_"))
        if not files:
            raise FileNotFoundError(f"no emotion experts in {emotion_dir}")
        self.emotion_names = [pretty_name(p.stem) for p in files]
        pws, pbs = [], []
        w2, b2, w3, b3, w4, b4, w5, b5 = [], [], [], [], [], [], [], []
        for p in files:
            sd = load_state(p)
            pws.append(sd["proj.weight"].to(dtype).to(device))
            pbs.append(sd["proj.bias"].to(dtype).to(device))
            w2.append(sd["mlp.2.weight"].to(device)); b2.append(sd["mlp.2.bias"].to(device))
            w3.append(sd["mlp.5.weight"].to(device)); b3.append(sd["mlp.5.bias"].to(device))
            w4.append(sd["mlp.8.weight"].to(device)); b4.append(sd["mlp.8.bias"].to(device))
            w5.append(sd["mlp.11.weight"].to(device)); b5.append(sd["mlp.11.bias"].to(device))
        self.pw, self.pb = pws, pbs
        self.mlp = (w2, b2, w3, b3, w4, b4, w5, b5)
        self.n_emotion = len(files)

        # --- quality experts ------------------------------------------------
        self.quality_names = list(QUALITY_FILES)
        qw, qb = [], []
        q2, qb2, q3, qb3, q4, qb4, q5, qb5 = [], [], [], [], [], [], [], []
        for name in self.quality_names:
            sd = load_state(plus_dir / QUALITY_FILES[name])
            qw.append(sd["proj.weight"].to(device)); qb.append(sd["proj.bias"].to(device))
            q2.append(sd["mlp.2.weight"].to(device)); qb2.append(sd["mlp.2.bias"].to(device))
            q3.append(sd["mlp.5.weight"].to(device)); qb3.append(sd["mlp.5.bias"].to(device))
            q4.append(sd["mlp.8.weight"].to(device)); qb4.append(sd["mlp.8.bias"].to(device))
            q5.append(sd["mlp.11.weight"].to(device)); qb5.append(sd["mlp.11.bias"].to(device))
        self.qw, self.qb = qw, qb
        self.qmlp = (q2, qb2, q3, qb3, q4, qb4, q5, qb5)
        print(f"EmpathicScorer: {self.n_emotion} emotion experts + "
              f"{len(self.quality_names)} quality experts")

    # ------------------------------------------------------------------
    @staticmethod
    def load_audio(path: str | Path, max_s: float = 30.0) -> np.ndarray:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != C.SR:
            wav = soxr.resample(wav, sr, C.SR, quality="HQ")
        n = int(max_s * C.SR)
        if len(wav) > n:
            wav = wav[:n]
        elif len(wav) < n:
            wav = np.pad(wav, (0, n - len(wav)))
        return np.ascontiguousarray(wav, dtype=np.float32)

    def _mlp_forward(self, h, mlp):
        w2, b2, w3, b3, w4, b4, w5, b5 = mlp
        h = F.relu(h)
        h = F.relu(F.linear(h, w2, b2))
        h = F.relu(F.linear(h, w3, b3))
        h = F.relu(F.linear(h, w4, b4))
        return F.linear(h, w5, b5)

    @torch.inference_mode()
    def score_batch(self, paths: list[str | Path]) -> list[dict]:
        wavs = [self.load_audio(p) for p in paths]
        inputs = self.fe(wavs, sampling_rate=C.SR, return_tensors="pt")
        feats = inputs.input_features.to(self.device, dtype=self.dtype)
        emb = self.encoder(feats).last_hidden_state  # [B,1500,768]
        flat = emb.reshape(emb.shape[0], -1).float()  # [B,1152000]
        flat16 = flat.half()
        emb_f = emb.float()
        pooled = torch.cat([emb_f.mean(1), emb_f.amin(1),
                            emb_f.amax(1), emb_f.std(1)], dim=1)

        emo = []
        for e in range(self.n_emotion):
            h = F.linear(flat16, self.pw[e], self.pb[e]).float()
            emo.append(self._mlp_forward(h, tuple(x[e] for x in self.mlp)))
        emo = torch.cat(emo, dim=1).float().cpu().numpy()  # [B,54]

        qual = []
        for e in range(len(self.quality_names)):
            h = F.linear(pooled, self.qw[e].float(), self.qb[e].float())
            qual.append(self._mlp_forward(h, tuple(x[e] for x in self.qmlp)))
        qual = torch.cat(qual, dim=1).float().cpu().numpy()  # [B,4]

        out = []
        for i in range(len(paths)):
            row = {name: round(float(v), 4)
                   for name, v in zip(self.emotion_names, emo[i])}
            row.update({name: round(float(v), 4)
                        for name, v in zip(self.quality_names, qual[i])})
            # Top-5 emotion labels (exclude demographic/quality axes).
            skip = {"Age", "Gender", "High-Pitched vs Low-Pitched",
                    "Recording Quality", "Background Noise"}
            cand = [(k, v) for k, v in row.items()
                    if k not in skip and k not in QUALITY_FILES]
            cand.sort(key=lambda kv: kv[1], reverse=True)
            row["_top5"] = [{"label": k, "score": v} for k, v in cand[:5]]
            out.append(row)
        return out

    def score_paths(self, paths, batch_size: int = 32, verbose: bool = True,
                    progress_every: int = 20) -> list[dict]:
        results = []
        for i in range(0, len(paths), batch_size):
            results.extend(self.score_batch(paths[i:i + batch_size]))
            if verbose and ((i // batch_size) % progress_every == 0):
                print(f"  empathic {i + len(paths[i:i+batch_size])}/{len(paths)}", flush=True)
        return results


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    sc = EmpathicScorer()
    for p, r in zip(args.audio, sc.score_paths(args.audio, args.batch_size)):
        print(p, json.dumps({k: v for k, v in r.items() if k != "_top5"}, indent=0)[:800])
