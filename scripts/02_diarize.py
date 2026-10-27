"""02 - Speaker diarization with NVIDIA Nemotron 3 Diarization.

Reads  manifests/files.csv  ->  outputs/diarization/<file_id>.json
Resumable: existing valid JSON files are skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from nemotron_diarize import (SR, diarize_chunked, diarize_streaming,
                              load_audio_16k, load_model)


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


def valid(path: Path) -> bool:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return bool(d.get("segments")) and d.get("duration", 0) > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mode", default="offline",
                    choices=["offline", "low_latency", "very_low_latency", "ultra_low_latency"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = load_manifest(args.ids)
    rows = [r for r in rows if args.overwrite or not valid(C.DIAR / f"{r['file_id']}.json")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"to diarize: {len(rows)}")
    if not rows:
        return

    t0 = time.time()
    processor, model = load_model(args.device)
    print(f"model loaded in {time.time()-t0:.1f}s")
    audio_s = 0.0
    for i, r in enumerate(rows, 1):
        fid = r["file_id"]
        out = C.DIAR / f"{fid}.json"
        if out.exists() and not args.overwrite:
            continue
        src = Path(r["path"])
        try:
            t = time.time()
            wav = load_audio_16k(src)
            dur = len(wav) / SR
            if args.mode == "offline":
                segs, _, n = diarize_chunked(processor, model, wav, device=args.device)
            else:
                segs, _, n = diarize_streaming(processor, model, wav, args.mode, args.device)
            dt = time.time() - t
            audio_s += dur
            payload = {
                "file_id": fid,
                "audio": str(src),
                "duration": dur,
                "sample_rate": SR,
                "model": C.NEMOTRON_ID,
                "mode": args.mode,
                "inference_seconds": round(dt, 3),
                "rtfx": round(dur / dt, 2),
                "n_chunks": n,
                "mel_rtfx_total": round(audio_s / max(time.time() - t0, 1e-6), 1),
                "segments": segs,
            }
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(out)
            print(f"[{i}/{len(rows)}] {fid} {dur/60:.1f} min -> "
                  f"{len(segs)} segs, {dt:.1f}s (RTFx {dur/dt:.0f})", flush=True)
        except Exception:
            print(f"FAIL {fid}", flush=True)
            traceback.print_exc()
            err = out.with_suffix(".error.txt")
            err.write_text(traceback.format_exc(), encoding="utf-8")

    print(f"finished in {(time.time()-t0)/60:.1f} min; audio {audio_s/3600:.2f} h")


if __name__ == "__main__":
    main()
