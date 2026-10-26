"""06 - Score every speaker snippet with Empathic-Insight-Voice-Plus.

55 affect/attribute experts (Empathic-Insight-Voice-Small) + 4 audio-quality
experts (Empathic-Insight-Voice-Plus), fused on one BUD-E-Whisper encoder pass.

Output: outputs/scores/<file_id>.empathic.jsonl
        {"segment_id": ..., "empathic": {label: score, ...}, "empathic_top5": [...]}
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score_common as S
from empathic_scorer import EmpathicScorer

STAGE = "empathic"


def score_paths(scorer, paths, batch_size):
    out = []
    for rec in scorer.score_batch(paths):
        top5 = rec.pop("_top5", [])
        out.append({"empathic": rec, "empathic_top5": top5})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = S.load_manifest(args.ids, args.limit)
    print(f"files: {len(rows)}")

    try:
        S.run_stage(STAGE, lambda: EmpathicScorer(device=args.device), score_paths,
                    rows, batch_size=args.batch_size,
                    max_seconds=args.max_seconds, overwrite=args.overwrite)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
