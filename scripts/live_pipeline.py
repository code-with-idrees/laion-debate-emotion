"""live_pipeline.py - keep the GPU busy while the corpus is still downloading.

Stage 01 writes one manifest row per finished WAV, and every later stage is
resumable. This driver repeatedly runs the per-file stages over whatever is
already complete, so diarization, ASR and snippet cutting finish days before the
436 GB download does.

    python scripts/live_pipeline.py                     # until the corpus is done
    python scripts/live_pipeline.py --limit 40 --sleep 60
    python scripts/live_pipeline.py --stages diarize transcribe

It exits only when the manifest holds ``--expect`` finished recordings *and*
every selected stage is drained; while the download is still running it simply
sleeps when there is nothing new to do.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

HERE = Path(__file__).parent
PY = sys.executable

# stage -> (script, output used to decide whether a file is done, extra args)
STAGES = {
    "transcripts": ("03_load_transcripts.py",
                    lambda fid: (C.TRANSCRIPTS / f"{fid}.json").exists(), []),
    "diarize": ("02_diarize.py", lambda fid: (C.DIAR / f"{fid}.json").exists(), []),
    "turns": ("04_build_turns.py", lambda fid: (C.TURNS / f"{fid}.jsonl").exists(), []),
    "snippets": ("05_cut_snippets.py",
                 lambda fid: (C.SNIPPETS / f"{fid}.index.jsonl").exists(), []),
}

# A stage that only takes --limit is wrong for 03: it streams the whole
# published transcript file and skips what is already on disk.
NO_LIMIT = {"transcripts"}


def manifest_rows():
    p = C.MANIFESTS / "files.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r.get("status") in ("ok", "skip")]


def pending_counts(rows, stages):
    out = {}
    for name in stages:
        _, done, _ = STAGES[name]
        out[name] = sum(1 for r in rows if not done(r["file_id"]))
    return out


def run_stage(name, limit, jobs):
    script, _, extra = STAGES[name]
    cmd = [PY, "-u", str(HERE / script), *extra]
    if name in ("turns", "snippets"):
        cmd += ["--limit", str(limit), "--jobs", str(jobs)]
    elif name not in NO_LIMIT:
        cmd += ["--limit", str(limit)]
    print(f"\n>>> {name}: {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(HERE.parent))
    print(f"<<< {name} exit={rc} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["transcripts", "diarize", "turns", "snippets"])
    ap.add_argument("--limit", type=int, default=25,
                    help="recordings per stage invocation")
    ap.add_argument("--jobs", type=int, default=0,
                    help="CPU workers for turns/snippets (default: cores-1)")
    ap.add_argument("--sleep", type=int, default=120,
                    help="seconds to wait when nothing is ready")
    ap.add_argument("--expect", type=int, default=C.DATASET_N_FILES,
                    help="manifest rows that mean the download is complete")
    ap.add_argument("--max-rounds", type=int, default=0, help="0 = unlimited")
    args = ap.parse_args()

    bad = [s for s in args.stages if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stages {bad}; choose from {list(STAGES)}")
    jobs = args.jobs or max(1, (C.N_CPU or 4) - 1)

    rounds = 0
    failures = 0
    t_start = time.time()
    while True:
        rounds += 1
        rows = manifest_rows()
        pending = pending_counts(rows, args.stages)
        total_pending = sum(pending.values())
        busy = {k: v for k, v in pending.items() if v}
        print(f"[{time.time() - t_start:8.0f}s] manifest={len(rows)}/{args.expect} "
              f"pending={busy or 'none'}", flush=True)

        if not busy and len(rows) >= args.expect:
            print("manifest complete and every stage drained; stopping.", flush=True)
            return 0

        for name in args.stages:
            if pending[name] <= 0:
                continue
            rc = run_stage(name, args.limit, jobs)
            if rc != 0:
                failures += 1
                print(f"stage {name} returned {rc}; backing off", flush=True)
                time.sleep(args.sleep)
                break

        if args.max_rounds and rounds >= args.max_rounds:
            print(f"stopped after {rounds} rounds ({failures} failures)", flush=True)
            return 0
        if not busy:
            time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
