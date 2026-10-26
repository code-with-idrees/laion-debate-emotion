"""VoiceNet dimension scorer (batched, 57 regressions + 57 classifications)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
import torch.nn.functional as F
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).parent))
import config as C

TARGET = 30 * C.SR


class MLPHead(torch.nn.Module):
    def __init__(self, D, H, p, out):
        super().__init__()
        self.f1 = torch.nn.Linear(D, H)
        self.f2 = torch.nn.Linear(H, out)

    def forward(self, x):
        return self.f2(F.gelu(self.f1(x)))


class VoiceNetScorer:
    def __init__(self, root: Path | None = None, device: str = "cuda",
                 variant: str | None = None):
        root = Path(root or C.VOICENET_DIR)
        src = root / "voiceclap_commercial"
        self.device = device
        self.embedder = AutoModel.from_pretrained(str(src), trust_remote_code=True)
        self.embedder.to(device).eval()
        base = root / "variants" / variant if variant else root
        self.reg, self.cls = {}, {}
        for p in sorted(glob.glob(str(base / "regression" / "*.pt"))):
            self.reg[Path(p).stem] = self._load_head(p)
        for p in sorted(glob.glob(str(base / "classification" / "*.pt"))):
            self.cls[Path(p).stem] = self._load_head(p)
        self.dims = sorted(self.reg)
        self.meta = {d: {"name": self.cls[d]["name"],
                         "n_levels": int(self.cls[d]["arch"]["out"]),
                         "levels": self.cls[d].get("levels", {})}
                     for d in self.dims}
        print(f"VoiceNetScorer: {len(self.dims)} dimensions")

    def _load_head(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck["arch"]
        net = MLPHead(a["D"], a["H"], a["p"], a["out"]).to(self.device).eval()
        net.load_state_dict(ck["state_dict"])
        mu = torch.as_tensor(np.asarray(ck["mu"]), dtype=torch.float32, device=self.device)
        sd = torch.as_tensor(np.asarray(ck["sd"]), dtype=torch.float32, device=self.device)
        return {"net": net, "mu": mu, "sd": sd, "arch": a,
                "name": ck.get("name", path), "levels": ck.get("levels", {})}

    @staticmethod
    def load_audio(path: str | Path) -> np.ndarray:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != C.SR:
            wav = soxr.resample(wav, sr, C.SR, quality="HQ")
        if len(wav) > TARGET:
            wav = wav[:TARGET]
        elif len(wav) < TARGET:
            wav = np.pad(wav, (0, TARGET - len(wav)))
        return np.ascontiguousarray(wav, dtype=np.float32)

    @torch.inference_mode()
    def score_batch(self, paths) -> list[dict]:
        wavs = np.stack([self.load_audio(p) for p in paths])
        x = torch.from_numpy(wavs).to(self.device)
        emb = self.embedder.encode_waveform(x).float()  # [B,768] L2-normalised

        reg_out, cls_out, conf_out = {}, {}, {}
        for d in self.dims:
            r = self.reg[d]
            xr = (emb - r["mu"]) / r["sd"]
            reg_out[d] = r["net"](xr).squeeze(-1)
            c = self.cls[d]
            xc = (emb - c["mu"]) / c["sd"]
            prob = torch.softmax(c["net"](xc), dim=-1)
            conf, bucket = prob.max(dim=-1)
            cls_out[d] = bucket
            conf_out[d] = conf

        out = []
        for i in range(len(paths)):
            out.append({
                "reg": {d: round(float(reg_out[d][i]), 4) for d in self.dims},
                "cls": {d: int(cls_out[d][i]) for d in self.dims},
                "conf": {d: round(float(conf_out[d][i]), 4) for d in self.dims},
            })
        return out

    def level_map(self) -> dict:
        """dim -> {name, n_levels, levels} for the bucket-label sidecar."""
        return {d: {"dim": d, "name": self.meta[d]["name"],
                    "n_levels": self.meta[d]["n_levels"],
                    "levels": {str(k): v for k, v in
                               (self.meta[d].get("levels") or {}).items()}}
                for d in self.dims}

    def score_paths(self, paths, batch_size: int = 32, verbose: bool = True):
        res = []
        for i in range(0, len(paths), batch_size):
            res.extend(self.score_batch(paths[i:i + batch_size]))
            if verbose and (i // batch_size) % 20 == 0:
                print(f"  voicenet {i + len(paths[i:i+batch_size])}/{len(paths)}", flush=True)
        return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    args = ap.parse_args()
    sc = VoiceNetScorer()
    for p, r in zip(args.audio, sc.score_paths(args.audio, 4)):
        print(p, json.dumps(r["reg"], indent=0)[:500])
