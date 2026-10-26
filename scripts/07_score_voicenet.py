"""07 - Score every speaker snippet with VoiceNet dimension predictors.

57 regression heads + 57 classification heads on one voiceclap-commercial
embedding (768-d) per snippet.

Output: outputs/scores/<file_id>.voicenet.jsonl
        {"segment_id": ..., "voicenet": {"reg": {...}, "cls": {...},
                                         "conf": {...}, "labels": {...}}}
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import score_common as S
from voicenet_scorer import VoiceNetScorer

STAGE = "voicenet"


def score_paths(scorer, paths, batch_size):
    return [{"voicenet": rec} for rec in scorer.score_batch(paths)]


def make_scorer(device, variant):
    sc = VoiceNetScorer(device=device, variant=variant)
    # Bucket descriptions are long prose; keep them once in a sidecar and store
    # only bucket ids + confidences per row.
    (C.DATASET / "voicenet_levels.json").write_text(
        json.dumps(sc.level_map(), indent=2), encoding="utf-8")
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default=None,
                    help="cap3x|cap4x|cap5x, default = best per dimension")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = S.load_manifest(args.ids, args.limit)
    print(f"files: {len(rows)}")

    try:
        S.run_stage(STAGE, lambda: make_scorer(args.device, args.variant),
                    score_paths, rows, batch_size=args.batch_size,
                    max_seconds=args.max_seconds, overwrite=args.overwrite)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
