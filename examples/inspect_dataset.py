"""Inspect the assembled dataset without loading the whole table.

    python examples/inspect_dataset.py                  # summary
    python examples/inspect_dataset.py --row 42         # one row as JSON
    python examples/inspect_dataset.py --row 42 --embedding
                                                        # + its VoiceCLAP vector

Works on the default output path and on any table produced with
``09_assemble_dataset.py --out``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def resolve(p: str | None) -> Path:
    p = Path(p) if p else ROOT / "dataset" / "data" / "train-00000-of-00001.parquet"
    return p if p.is_absolute() else ROOT / p


def to_jsonable(o):
    if isinstance(o, dict):
        return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return o


def summarize(pf, table):
    n = table.num_rows
    print(f"rows: {n:,}")
    print(f"columns: {len(table.column_names)}")
    print(f"row groups: {pf.num_row_groups}")
    print()
    for name in table.column_names:
        col = table.column(name)
        nn = col.null_count
        print(f"  {name:<22} {str(col.type):<40} nulls {nn:>9,}")
    if n == 0:
        return

    d = table.to_pydict()
    print()
    print(f"files:     {len(set(d['file_id'])):,}")
    print(f"speakers:  {len({(f, s) for f, s in zip(d['file_id'], d['speaker']) if s}):,}")
    print(f"duration:  {sum(d['duration']):,.1f} s")
    if d.get("empathic_top5") and d["empathic_top5"][0]:
        labels = Counter()
        for top in d["empathic_top5"]:
            if top:
                labels[top[0]["label"]] += 1
        print("\ntop-1 empathic label:")
        for label, c in labels.most_common(10):
            print(f"  {c / n:6.1%}  {label}")


def show_row(table, idx: int, show_embedding: bool):
    if not 0 <= idx < table.num_rows:
        sys.exit(f"row {idx} out of range (0..{table.num_rows - 1})")
    row = to_jsonable({k: table.column(k)[idx].as_py() for k in table.column_names})
    print(json.dumps(row, indent=2, ensure_ascii=False))

    if not show_embedding:
        return
    emb_file = row.get("voiceclap_emb_file")
    emb_row = row.get("voiceclap_emb_row")
    if not emb_file or emb_row is None:
        print("\n(no VoiceCLAP embedding for this row)")
        return
    path = resolve(emb_file)
    if not path.exists():
        print(f"\n(embedding sidecar missing: {path})")
        return
    mat = np.load(path)
    vec = mat[emb_row]
    print(f"\nembedding: {path.name}[{emb_row}] shape={vec.shape} "
          f"dtype={mat.dtype} L2={np.linalg.norm(vec):.4f}")
    print(f"min={vec.min():.4f} max={vec.max():.4f} mean={vec.mean():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--row", type=int, default=None,
                    help="print one row (0-based index across the whole table)")
    ap.add_argument("--embedding", action="store_true",
                    help="with --row, also load the referenced VoiceCLAP vector")
    args = ap.parse_args()

    path = resolve(args.parquet)
    if not path.exists():
        sys.exit(f"no parquet at {path}; run scripts/09_assemble_dataset.py first")
    table = pq.read_table(path)
    if args.row is None:
        summarize(pq.ParquetFile(path), table)
    else:
        show_row(table, args.row, args.embedding)


if __name__ == "__main__":
    main()
