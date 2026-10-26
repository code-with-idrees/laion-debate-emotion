"""05 - Cut one 16 kHz mono FLAC snippet per speaker turn.

Inputs : outputs/turns/<id>.jsonl + data/audio/<id>.wav
Output : outputs/snippets/<id>/<segment_id>.flac
         outputs/snippets/<id>.index.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from parallel import run_rows

PAD_S = 0.05


def load_manifest(ids=None, limit=0):
    p = C.MANIFESTS / "files.csv"
    with p.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") in ("ok", "skip")]
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["file_id"] in wanted]
    if limit:
        rows = rows[:limit]
    return rows


def cut_file(row, overwrite=False):
    fid = row["file_id"]
    turns_path = C.TURNS / f"{fid}.jsonl"
    if not turns_path.exists():
        return {"file_id": fid, "status": "missing_turns"}
    outdir = C.SNIPPETS / fid
    index_path = C.SNIPPETS / f"{fid}.index.jsonl"
    if index_path.exists() and not overwrite:
        return None
    outdir.mkdir(parents=True, exist_ok=True)

    audio_path = Path(row.get("path") or C.AUDIO / f"{fid}.wav")
    if not audio_path.exists():
        return {"file_id": fid, "status": "missing_audio"}

    turns = [json.loads(l) for l in turns_path.open(encoding="utf-8")]
    rows = []
    with sf.SoundFile(str(audio_path)) as fh:
        sr = fh.samplerate
        n = len(fh)
        for t in turns:
            seg_id = t["segment_id"]
            dest = outdir / f"{seg_id}.flac"
            a = max(0.0, float(t["start"]) - PAD_S)
            b = min(n / sr, float(t["end"]) + PAD_S)
            if b <= a:
                continue
            if dest.exists() and dest.stat().st_size > 0 and not overwrite:
                rows.append({"segment_id": seg_id, "file_id": fid,
                             "path": str(dest), "duration": round(b - a, 3)})
                continue
            fh.seek(int(a * sr))
            x = fh.read(int(round((b - a) * sr)), dtype="float32", always_2d=False)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != C.SR:
                x = soxr.resample(x, sr, C.SR, quality="HQ")
            x = np.clip(x, -1.0, 1.0).astype(np.float32)
            sf.write(str(dest), x, C.SR, format="FLAC", subtype="PCM_16")
            rows.append({"segment_id": seg_id, "file_id": fid,
                         "path": str(dest), "duration": round(len(x) / C.SR, 3)})

    tmp = index_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(index_path)
    total_s = sum(r["duration"] for r in rows)
    return {"file_id": fid, "status": "ok", "n": len(rows),
            "audio_s": round(total_s, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes (default: all cores)")
    args = ap.parse_args()

    rows = load_manifest(args.ids, args.limit)
    print(f"files: {len(rows)}", flush=True)
    for i, total, row, status, payload in run_rows(
            rows, cut_file, args.jobs, args.overwrite):
        fid = row["file_id"]
        if status == "error":
            print(f"FAIL {fid}", flush=True)
            print(payload, flush=True)
            (C.SNIPPETS / f"{fid}.error.txt").write_text(payload, encoding="utf-8")
            continue
        if payload:
            print(f"[{i}/{total}] {fid} -> {payload}", flush=True)


if __name__ == "__main__":
    main()
