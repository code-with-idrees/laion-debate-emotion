"""Print how many recordings still lack each model's scores.

Used by scripts/run_scores.ps1 to decide whether to run a stage and when the
build is finished. It lives in a file rather than an inline `python -c` because
Windows PowerShell 5.1 drops embedded double quotes when passing a multi-line
here-string to a native command, which made the old check silently return
nothing and run all three stages on every pass.

Only recordings that actually contain snippets count:
  * the diarization example fixture is not a corpus recording;
  * a handful of recordings align to zero turns, so their index is empty and no
    stage can ever score them. Counting those would keep the loop from ever
    deciding it is done.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

STAGES = ("voicenet", "empathic", "voiceclap")


def usable_with_snippets():
    with (C.MANIFESTS / "files.csv").open(encoding="utf-8") as fh:
        ok = {r["file_id"] for r in csv.DictReader(fh)
              if r.get("status") in ("ok", "skip")}
    out = set()
    for p in C.SNIPPETS.glob("*.index.jsonl"):
        fid = p.name[:-len(".index.jsonl")]
        if fid not in ok:
            continue
        with p.open(encoding="utf-8") as fh:
            if any(line.strip() for line in fh):
                out.add(fid)
    return out


def pending():
    idx = usable_with_snippets()
    counts = []
    for stage in STAGES:
        done = {p.name[:-len(f".{stage}.jsonl")]
                for p in C.SCORES.glob(f"*.{stage}.jsonl")} & idx
        counts.append(len(idx) - len(done))
    return counts


def main():
    print(*pending())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
