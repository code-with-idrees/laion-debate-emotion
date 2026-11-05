"""Verify the assembled dataset (script 09 output).

Checks the invariants the build is supposed to guarantee, so a finished Parquet
can be signed off without opening it by hand:

  * one row per (file_id, segment_id), with start < end and a speaker
  * every row has a duration consistent with its boundaries
  * the three requested model columns are present
  * every voiceclap_emb_file / voiceclap_emb_row pointer resolves to an existing
    .npy whose ids.json names that same segment (the classic silent corruption:
    a row index off by one after a failed snippet)

Usage:
  python scripts/verify_dataset.py
  python scripts/verify_dataset.py --parquet dataset/data/train-00000-of-00001.parquet
  python scripts/verify_dataset.py --sample 5000     # check a random subset of rows
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import config as C

EMB_REL = "outputs/embeddings/voiceclap"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(
        C.DATASET / "data" / "train-00000-of-00001.parquet"))
    ap.add_argument("--sample", type=int, default=0,
                    help="check this many random rows for embedding pointers (0 = all)")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/09_assemble_dataset.py")

    t = pq.read_table(path)
    rows = t.to_pylist()
    print(f"parquet   {path}")
    print(f"rows      {len(rows):,}   columns {t.num_columns}")
    if not rows:
        raise SystemExit("dataset is empty")

    problems: Counter = Counter()
    seen = set()
    for r in rows:
        key = (r["file_id"], r["segment_id"])
        if key in seen:
            problems["duplicate (file_id, segment_id)"] += 1
        seen.add(key)
        if r["end"] <= r["start"]:
            problems["end <= start"] += 1
        if abs((r["end"] - r["start"]) - r["duration"]) > 0.05:
            problems["duration != end - start"] += 1
        if not r["speaker"]:
            problems["missing speaker"] += 1
        if not r["text"]:
            problems["missing text"] += 1

    has = {
        "empathic": sum(1 for r in rows if r["empathic"] is not None),
        "voicenet": sum(1 for r in rows if r["voicenet"] is not None),
        "voiceclap": sum(1 for r in rows if r["voiceclap_top5"] is not None),
    }
    for k, n in has.items():
        print(f"{k:10s}{n:>9,} / {len(rows):,}  ({100 * n / len(rows):.2f}%)")

    # Embedding pointers, resolved through ids.json rather than assumed order.
    candidates = [r for r in rows if r["voiceclap_emb_file"]]
    if args.sample and len(candidates) > args.sample:
        candidates = random.Random(0).sample(candidates, args.sample)
    cache = {}
    for r in candidates:
        fid = r["file_id"]
        if fid not in cache:
            ids_p = Path(EMB_REL) / f"{fid}.ids.json"
            npy_p = Path(EMB_REL) / f"{fid}.npy"
            if not (ids_p.exists() and npy_p.exists()):
                cache[fid] = None
            else:
                ids = json.loads(ids_p.read_text(encoding="utf-8"))
                n = np.load(npy_p, mmap_mode="r").shape[0]
                cache[fid] = ids if len(ids) == n else None
        ids = cache[fid]
        if ids is None:
            problems["embedding file/ids.json missing or length mismatch"] += 1
            continue
        j = r["voiceclap_emb_row"]
        if j is None or j < 0 or j >= len(ids):
            problems["embedding row out of range"] += 1
        elif ids[j] != r["segment_id"]:
            problems["embedding row points at another segment"] += 1

    print(f"embedding pointers checked  {len(candidates):,}")
    print(f"speakers  {len({r['speaker'] for r in rows})}   "
          f"files  {len({r['file_id'] for r in rows})}   "
          f"speech hours  {sum(r['duration'] for r in rows) / 3600:.2f}")

    if problems:
        print("\nFAIL")
        for k, n in problems.most_common():
            print(f"  {n:>9,}  {k}")
        raise SystemExit(1)

    print("\nOK - no invariant violations")


if __name__ == "__main__":
    main()
