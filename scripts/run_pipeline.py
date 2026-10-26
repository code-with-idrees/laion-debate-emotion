"""run_pipeline.py - drive every stage end to end.

Each stage runs as its own subprocess so GPU memory is released between models.

  python run_pipeline.py                     # everything, resumable
  python run_pipeline.py --from snippets      # resume from a stage
  python run_pipeline.py --stages diarize transcribe
  python run_pipeline.py --limit 5 --dry-run  # smoke test on 5 files
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable

# name -> (script, extra args)
STAGES = [
    ("download", "01_download_dataset.py", []),
    ("diarize", "02_diarize.py", []),
    ("transcribe", "03_transcribe.py", []),
    ("turns", "04_build_turns.py", []),
    ("snippets", "05_cut_snippets.py", []),
    ("empathic", "06_score_empathic.py", []),
    ("voicenet", "07_score_voicenet.py", []),
    ("voiceclap", "08_score_voiceclap.py", []),
    ("assemble", "09_assemble_dataset.py", []),
]

TAKES_LIMIT = {"download", "diarize", "transcribe", "turns", "snippets",
               "empathic", "voicenet", "voiceclap", "assemble"}
TAKES_IDS = TAKES_LIMIT
TAKES_DEVICE = {"diarize", "empathic", "voicenet", "voiceclap"}
CPU_STAGES = {"turns", "snippets", "assemble"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--from", dest="from_stage", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4, help="download workers")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=0,
                    help="worker processes for CPU stages (default: all cores)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = [s[0] for s in STAGES]
    if args.stages:
        bad = [s for s in args.stages if s not in names]
        if bad:
            raise SystemExit(f"unknown stages {bad}; choose from {names}")
        selected = [s for s in STAGES if s[0] in args.stages]
    elif args.from_stage:
        if args.from_stage not in names:
            raise SystemExit(f"unknown stage {args.from_stage}; choose from {names}")
        selected = STAGES[names.index(args.from_stage):]
    else:
        selected = STAGES

    for name, script, extra in selected:
        cmd = [PY, "-u", str(HERE / script)]
        if args.limit and name in TAKES_LIMIT:
            cmd += ["--limit", str(args.limit)]
        if args.ids and name in TAKES_IDS:
            cmd += ["--ids", *args.ids]
        if name == "download" and args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.device and name in TAKES_DEVICE:
            cmd += ["--device", args.device]
        if args.batch_size and name in ("empathic", "voicenet", "voiceclap"):
            cmd += ["--batch-size", str(args.batch_size)]
        if args.jobs and name in CPU_STAGES:
            cmd += ["--jobs", str(args.jobs)]
        if args.jobs and name == "transcribe":
            cmd += ["--cpu-threads", str(args.jobs)]
        cmd += extra

        print(f"\n{'='*70}\n>>> {name}: {' '.join(cmd)}\n{'='*70}", flush=True)
        if args.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(HERE.parent))
        print(f"<<< {name} exit={rc} in {(time.time()-t0)/60:.1f} min", flush=True)
        if rc != 0:
            print(f"stage {name} failed; stopping. Re-run to resume.", flush=True)
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
