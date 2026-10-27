"""03 - Re-transcribe every recording with Whisper large-v3 (CTranslate2).

Reads  manifests/files.csv  ->  outputs/transcripts/<file_id>.json
Word-level timestamps are required for speaker attribution.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

from faster_whisper import BatchedInferencePipeline, WhisperModel


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
        return bool(d.get("segments"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--model", default=str(C.WHISPER_CT2))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--language", default="en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu-threads", type=int, default=0,
                    help="CTranslate2 CPU threads (default: all cores)")
    args = ap.parse_args()
    cpu_threads = args.cpu_threads or (os.cpu_count() or 4)

    rows = load_manifest(args.ids)
    rows = [r for r in rows if args.overwrite or not valid(C.TRANSCRIPTS / f"{r['file_id']}.json")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"to transcribe: {len(rows)}")
    if not rows:
        return

    t0 = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type="float16",
                         cpu_threads=cpu_threads, num_workers=1)
    pipe = BatchedInferencePipeline(model=model)
    print(f"model loaded in {time.time()-t0:.1f}s")

    audio_s = 0.0
    for i, r in enumerate(rows, 1):
        fid = r["file_id"]
        out = C.TRANSCRIPTS / f"{fid}.json"
        if out.exists() and not args.overwrite:
            continue
        src = Path(r["path"])
        try:
            t = time.time()
            segments, info = pipe.transcribe(
                str(src),
                batch_size=args.batch_size,
                language=args.language or None,
                task="transcribe",
                beam_size=args.beam_size,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False,
                without_timestamps=False,
            )
            segs = []
            for s in segments:
                segs.append({
                    "id": s.id,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "words": [
                        {"word": w.word,
                         "start": round(w.start, 3) if w.start is not None else None,
                         "end": round(w.end, 3) if w.end is not None else None,
                         "probability": w.probability}
                        for w in (s.words or [])
                    ],
                })
            dt = time.time() - t
            dur = float(info.duration or 0.0)
            audio_s += dur
            payload = {
                "file_id": fid,
                "audio": str(src),
                "duration": dur,
                "language": info.language,
                "language_probability": info.language_probability,
                "model": args.model,
                "beam_size": args.beam_size,
                "inference_seconds": round(dt, 3),
                "rtfx": round(dur / dt, 2) if dt else None,
                "segments": segs,
            }
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(out)
            print(f"[{i}/{len(rows)}] {fid} {dur/60:.1f} min -> "
                  f"{len(segs)} segs, {dt:.1f}s (RTFx {dur/dt:.0f})", flush=True)
        except Exception:
            print(f"FAIL {fid}", flush=True)
            traceback.print_exc()
            (out.with_suffix(".error.txt")).write_text(traceback.format_exc(), encoding="utf-8")

    print(f"finished in {(time.time()-t0)/60:.1f} min; audio {audio_s/3600:.2f} h")


if __name__ == "__main__":
    main()
