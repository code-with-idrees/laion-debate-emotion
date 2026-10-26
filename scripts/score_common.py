"""Shared helpers for the per-model scoring stages (06-08).

Every scoring stage works file-by-file:

  outputs/snippets/<fid>.index.jsonl   (segment_id, duration, path)
      -> outputs/scores/<fid>.<stage>.jsonl   (resumable, atomic rename)

A file is considered finished when its output JSONL exists, parses, and covers
exactly the segment ids of the snippet index.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C


# --------------------------------------------------------------------------- #
# manifest / snippet index
# --------------------------------------------------------------------------- #
def load_manifest(ids=None, limit=0):
    p = C.MANIFESTS / "files.csv"
    if not p.exists():
        raise SystemExit(f"missing manifest {p}; run 01_download_dataset.py")
    with p.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") in ("ok", "skip")]
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["file_id"] in wanted]
    if limit:
        rows = rows[:limit]
    return rows


def load_index(fid: str):
    p = C.SNIPPETS / f"{fid}.index.jsonl"
    if not p.exists():
        return None
    rows = []
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# output bookkeeping
# --------------------------------------------------------------------------- #
def out_path(stage: str, fid: str) -> Path:
    return C.SCORES / f"{fid}.{stage}.jsonl"


def existing_ids(path: Path):
    """Parse an existing JSONL and return the segment ids found (or None)."""
    if not path.exists():
        return None
    ids = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)["segment_id"])
    except Exception:
        return None
    return ids


def is_done(stage: str, fid: str, rows) -> bool:
    ids = existing_ids(out_path(stage, fid))
    if not ids:
        return False
    return ids == [r["segment_id"] for r in rows]


def atomic_write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# batching / OOM handling
# --------------------------------------------------------------------------- #
def make_batches(items, max_batch: int, max_seconds: float):
    """Group consecutive snippet rows into batches bounded by count and duration."""
    cur, dur = [], 0.0
    for it in items:
        d = float(it.get("duration") or 0.0)
        if cur and (len(cur) >= max_batch or dur + d > max_seconds):
            yield cur
            cur, dur = [], 0.0
        cur.append(it)
        dur += d
    if cur:
        yield cur


def score_with_retry(score_batch, items, depth: int = 0):
    """Call score_batch(items); on CUDA OOM split the batch in half and retry."""
    try:
        return score_batch(items)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(items) <= 1 or depth > 6:
            raise
        mid = len(items) // 2
        print(f"    OOM at batch {len(items)} -> retry as {mid}+{len(items)-mid}",
              flush=True)
        left = score_with_retry(score_batch, items[:mid], depth + 1)
        right = score_with_retry(score_batch, items[mid:], depth + 1)
        return left + right


# --------------------------------------------------------------------------- #
# generic per-file driver
# --------------------------------------------------------------------------- #
def run_stage(stage: str, make_scorer, score_paths, rows, *,
              batch_size: int = 16, max_seconds: float = 240.0,
              overwrite: bool = False, per_file=None):
    """Score every snippet of every manifest file.

    make_scorer()      -> loaded model (called once, lazily)
    score_paths(sc, paths, batch_size) -> list[dict] aligned with paths
    per_file(fid, rows, out_rows) -> optional extra artifact writing
    """
    scorer = None
    total_rows = 0
    for i, mrow in enumerate(rows, 1):
        fid = mrow["file_id"]
        idx = load_index(fid)
        if not idx:
            print(f"[{i}/{len(rows)}] {fid}: no snippets, skip", flush=True)
            continue
        if not overwrite and is_done(stage, fid, idx):
            total_rows += len(idx)
            continue

        if scorer is None:
            scorer = make_scorer()

        out_rows = []
        for batch in make_batches(idx, batch_size, max_seconds):
            paths = [b["path"] for b in batch]
            payloads = score_with_retry(
                lambda items: score_paths(scorer, [x["path"] for x in items],
                                          batch_size),
                batch)
            for b, payload in zip(batch, payloads):
                rec = {"segment_id": b["segment_id"],
                       "duration": b.get("duration")}
                rec.update(payload)
                out_rows.append(rec)

        atomic_write_jsonl(out_path(stage, fid), out_rows)
        if per_file:
            per_file(fid, idx, out_rows)
        total_rows += len(out_rows)
        print(f"[{i}/{len(rows)}] {fid}: {len(out_rows)} segments scored", flush=True)
    print(f"stage {stage}: {total_rows} segment rows total")
    return total_rows
