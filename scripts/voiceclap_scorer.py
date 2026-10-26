"""VoiceCLAP-large-v2 audio/text scorer (sentence-transformers).

The model is a rank-16 LoRA finetune of LCO-Embedding-Omni-7B (Qwen2.5-Omni
thinker) trained contrastively, so audio and text live in the same 3584-d
space.  We encode the audio once per snippet, keep the full embedding, and
derive a compact top-k label list against a fixed prompt vocabulary.

Embeddings live in float16 .npy sidecars (one per source file); the tabular
output only carries the top-5 text probes plus the row index into the .npy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def resolve_model_path() -> str:
    """Prefer the local snapshot, fall back to the hub id."""
    local = Path(C.VOICECLAP_DIR)
    if (local / "modules.json").exists():
        return str(local)
    return C.VOICECLAP_ID


class VoiceCLAPScorer:
    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 dtype: str = "bfloat16", prompts=None, top_k: int = 5,
                 quant: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.device = device
        self.top_k = top_k
        self.prompts = list(prompts or C.VOICECLAP_PROMPTS)
        model_kwargs = {"torch_dtype": DTYPES[dtype], "low_cpu_mem_usage": True}
        if quant == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif quant == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=DTYPES[dtype],
                bnb_4bit_quant_type="nf4")
        self.model = SentenceTransformer(
            model_path or resolve_model_path(),
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            device=device,
        )
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()
        self.text_emb = None

    # ------------------------------------------------------------------ #
    def _cache_paths(self):
        return (C.EMBEDDINGS / "voiceclap_text_emb.npy",
                C.EMBEDDINGS / "voiceclap_text_prompts.json")

    def text_embeddings(self, force: bool = False) -> np.ndarray:
        if self.text_emb is not None and not force:
            return self.text_emb
        emb_p, prompts_p = self._cache_paths()
        if emb_p.exists() and prompts_p.exists() and not force:
            try:
                if json.loads(prompts_p.read_text(encoding="utf-8")) == self.prompts:
                    self.text_emb = np.load(emb_p)
                    if self.dim is None:
                        self.dim = int(self.text_emb.shape[1])
                    return self.text_emb
            except Exception:
                pass
        print(f"encoding {len(self.prompts)} text prompts ...", flush=True)
        emb = self.model.encode(self.prompts, batch_size=8, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        C.EMBEDDINGS.mkdir(parents=True, exist_ok=True)
        np.save(emb_p, emb)
        prompts_p.write_text(json.dumps(self.prompts), encoding="utf-8")
        self.text_emb = emb
        if self.dim is None:
            self.dim = int(emb.shape[1])
        return emb

    # ------------------------------------------------------------------ #
    def embed_audio(self, paths) -> np.ndarray:
        emb = self.model.encode([str(p) for p in paths],
                                batch_size=max(1, len(paths)),
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

    @torch.inference_mode()
    def score_batch(self, paths) -> tuple[list[dict], np.ndarray]:
        """Return (per-snippet payloads, float16 embeddings [B, D])."""
        emb = self.embed_audio(paths)
        if self.text_emb is None:
            self.text_emb = self.text_embeddings()
        sims = emb @ self.text_emb.T  # [B, P] cosine similarity

        out = []
        for i in range(len(paths)):
            order = np.argsort(-sims[i])[: self.top_k]
            top = [{"label": self.prompts[int(j)], "sim": round(float(sims[i][j]), 4)}
                   for j in order]
            out.append({"voiceclap_top5": top, "voiceclap_best": top[0]["label"],
                        "voiceclap_best_sim": top[0]["sim"]})
        return out, emb.astype(np.float16)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    args = ap.parse_args()
    sc = VoiceCLAPScorer(dtype=args.dtype)
    rows, emb = sc.score_batch(args.audio)
    print("dim", emb.shape)
    for p, r in zip(args.audio, rows):
        print(p, r["voiceclap_top5"][:3])
