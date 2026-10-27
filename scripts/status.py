"""status.py - progress report for every pipeline stage."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def count_files(d: Path, pattern: str) -> int:
    return sum(1 for _ in d.glob(pattern)) if d.exists() else 0


def count_for_ids(d: Path, pattern: str, ids: set[str], suffix: str) -> int:
    """Count outputs that belong to manifest rows.

    Transcripts are built for the whole corpus in one pass, so a raw file count
    reads as 20,000% while the download is still at 14 recordings.
    """
    if not d.exists():
        return 0
    return sum(1 for fid in ids if (d / f"{fid}{suffix}").exists())


def manifest_ids():
    p = C.MANIFESTS / "files.csv"
    if not p.exists():
        return [], []
    with p.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ok = [r for r in rows if r.get("status") in ("ok", "skip")]
    return ok, rows


def dir_size_gb(d: Path) -> float:
    if not d.exists():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e9


def bar(done: int, total: int, width: int = 24) -> str:
    if not total:
        return " " * width
    n = max(0, min(width, int(width * done / total)))
    return "#" * n + "." * (width - n)


def main():
    ok, all_rows = manifest_ids()
    n = len(ok)
    ids = {r["file_id"] for r in ok}
    print(f"root        {C.ROOT}")
    print(f"dataset     {C.DATASET_REPO}")
    print(f"manifest    {len(all_rows)} rows, {n} usable")
    if all_rows:
        errs = [r for r in all_rows if r.get("status") == "error"]
        print(f"errors      {len(errs)}")

    stages = [
        ("diarization", C.DIAR, "*.json", ".json"),
        ("transcripts", C.TRANSCRIPTS, "*.json", ".json"),
        ("turns", C.TURNS, "*.jsonl", ".jsonl"),
        ("snippets", C.SNIPPETS, "*.index.jsonl", ".index.jsonl"),
        ("empathic", C.SCORES, "*.empathic.jsonl", ".empathic.jsonl"),
        ("voicenet", C.SCORES, "*.voicenet.jsonl", ".voicenet.jsonl"),
        ("voiceclap", C.SCORES, "*.voiceclap.jsonl", ".voiceclap.jsonl"),
    ]
    for name, d, pat, suffix in stages:
        c = count_for_ids(d, pat, ids, suffix)
        pct = f"{min(100.0, 100 * c / n):5.1f}%" if n else "  n/a"
        print(f"{name:12s} {c:6d}/{n:<6d} {pct} [{bar(c, n)}] {dir_size_gb(d):8.2f} GB")

    skipped = count_files(C.TURNS, "*.skip.json")
    if skipped:
        print(f"{'skipped':12s} {skipped:6d} recordings with no usable transcript")

    emb = C.EMBEDDINGS / "voiceclap"
    ne = count_for_ids(emb, "*.npy", ids, ".npy")
    print(f"{'embeddings':12s} {ne:6d}/{n:<6d}       [{bar(ne, n)}] {dir_size_gb(emb):8.2f} GB")

    pq_file = C.DATASET / "data" / "train-00000-of-00001.parquet"
    if pq_file.exists():
        st = C.DATASET / "stats.json"
        info = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
        print(f"\ndataset     {pq_file.stat().st_size/1e9:.2f} GB, "
              f"rows={info.get('rows')}, speech_h={info.get('speech_hours')}")
        print(f"coverage    {info.get('coverage')}")
    else:
        print("\ndataset     not assembled yet")


if __name__ == "__main__":
    main()
