"""08 - Score every speaker snippet with VoiceCLAP-large-v2.

One 7B forward pass per snippet yields a 3584-d audio embedding in the same
space as the text probe vocabulary.  We keep
  * the full embedding  -> outputs/embeddings/voiceclap/<file_id>.npy  (fp16)
  * a compact top-5     -> outputs/scores/<file_id>.voiceclap.jsonl
plus the text embedding matrix in outputs/embeddings/voiceclap_text_emb.npy.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import score_common as S
from voiceclap_scorer import VoiceCLAPScorer

STAGE = "voiceclap"
EMB_DIR = C.EMBEDDINGS / "voiceclap"

_buffer: dict[str, list[tuple[str, np.ndarray]]] = {}


def score_paths(scorer, paths, batch_size):
    payloads, emb = scorer.score_batch(paths)
    fid = Path(paths[0]).parent.name
    # Keep each embedding next to the segment it belongs to. Storing the batch
    # alone loses that pairing: a snippet that fails to decode still produces an
    # error row but no embedding, so counting rows would shift every later row
    # index in the .npy and point the Parquet at the wrong vector.
    buf = _buffer.setdefault(fid, [])
    for path, vec in zip(paths, emb):
        buf.append((Path(path).stem, vec))
    return payloads


def per_file(fid, idx, out_rows):
    pairs = _buffer.pop(fid, None)
    if not pairs:
        return
    emb = np.vstack([v for _, v in pairs]).astype(np.float16)
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{fid}.npy", emb)
    # ids.json is the authoritative row order of the .npy: one entry per
    # embedding actually stored, not one per scored turn.
    ids = [sid for sid, _ in pairs]
    (EMB_DIR / f"{fid}.ids.json").write_text(json.dumps(ids), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant", default=None, choices=[None, "8bit", "4bit"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = S.load_manifest(args.ids, args.limit)
    print(f"files: {len(rows)}")

    def make():
        sc = VoiceCLAPScorer(device=args.device, dtype=args.dtype,
                             top_k=args.top_k, quant=args.quant)
        sc.text_embeddings()
        print(f"VoiceCLAPScorer ready: dim={sc.dim}, prompts={len(sc.prompts)}")
        return sc

    try:
        S.run_stage(STAGE, make, score_paths, rows,
                    batch_size=args.batch_size, max_seconds=args.max_seconds,
                    overwrite=args.overwrite, per_file=per_file)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
