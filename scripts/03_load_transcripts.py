"""03 - Use the corpus transcripts instead of re-running ASR.

`teamsleeping/sleeping-debate-transcribe` publishes `transcribe.jsonl`: one
object per recording with the full text and word-level timestamps (736 MB,
2,890 records). This stage streams that file and writes
`outputs/transcripts/<id>.json` in exactly the shape `04_build_turns.py`
already consumes, so speaker attribution and snippet cutting are unchanged.

Why not Whisper: the corpus ships its own transcripts, so re-transcribing 436 GB
buys nothing except days of GPU time and a second, disagreeing text. `03_
transcribe.py` stays available as an explicit fallback (`--whisper` in
run_pipeline) for recordings the published file does not cover.

Examples
--------
  python 03_load_transcripts.py
  python 03_load_transcripts.py --limit 50
  python 03_load_transcripts.py --ids PaKRFdAY-sY PWwTlDhd5OA
  python 03_load_transcripts.py --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

# A new segment starts after a pause this long, or when one gets this long.
SEG_GAP_S = 0.4
SEG_MAX_S = 12.0


def load_manifest(ids=None):
    p = C.MANIFESTS / "files.csv"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("status") in ("ok", "skip")]
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["file_id"] in wanted]
    return rows


def remote_size(path: str = "transcribe.jsonl") -> int:
    """Exact byte size from the repo listing.

    A HEAD request is not reliable here (redirect chain without content-length),
    and a zero size would make the downloader treat an empty placeholder as a
    finished file.
    """
    from huggingface_hub import HfApi

    info = HfApi(token=os.environ.get("HF_TOKEN")).dataset_info(
        C.TRANSCRIPT_REPO, files_metadata=True,
        token=os.environ.get("HF_TOKEN"))
    for s in info.siblings:
        if s.rfilename == path:
            return int(s.size or 0)
    return 0


def ensure_jsonl() -> Path:
    """Download the published transcripts once and return the local path."""
    src = C.TRANSCRIPT_JSONL
    if src.exists() and src.stat().st_size > 1_000_000:
        return src
    from hf_fetch import Downloader

    src.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and src.stat().st_size == 0:
        src.unlink()
    size = remote_size()
    if not size:
        raise SystemExit(f"cannot resolve the size of {C.TRANSCRIPT_REPO}"
                         "/transcribe.jsonl")
    print(f"downloading {C.TRANSCRIPT_REPO}/transcribe.jsonl "
          f"({size / 1e9:.2f} GB) -> {src}", flush=True)
    dl = Downloader(os.environ.get("HF_TOKEN"), chunk_threads=4)
    ok, msg = dl.download(C.TRANSCRIPT_REPO, "transcribe.jsonl", src, size,
                          repo_type="dataset")
    if not ok:
        raise SystemExit(f"transcript download failed: {msg}")
    return src


def normalise_words(word_timestamps):
    """Whisper-style words: timings float, one leading space per word.

    The published file has bare words ("It", "gives"), while 04 rebuilds turn
    text with ``"".join(w["word"])`` the way Whisper emits it (leading spaces,
    except at a segment start). Re-adding the space keeps that join exact and
    leaves the stored ``word`` values identical in shape to a Whisper run.
    """
    out = []
    for i, w in enumerate(word_timestamps or []):
        text = w.get("word")
        if not text or not str(text).strip():
            continue
        start, end = w.get("start"), w.get("end")
        if start is None and end is None:
            continue
        if start is None:
            start = end
        if end is None:
            end = start
        start, end = float(start), float(end)
        if end < start:
            start, end = end, start
        text = str(text)
        if i and not text.startswith(" "):
            text = " " + text
        out.append({"word": text, "start": round(start, 3), "end": round(end, 3),
                    "probability": w.get("probability")})
    out.sort(key=lambda x: (x["start"], x["end"]))
    return out


def to_segments(words):
    """Group words into phrase-sized segments (pause or length based)."""
    segments, cur = [], []
    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            too_long = (w["end"] - cur[0]["start"]) > SEG_MAX_S
            # Epsilon: 10.64 - 10.24 is 0.4000000000000004 in binary floating
            # point, which would split a pause that is exactly SEG_GAP_S long.
            if gap > SEG_GAP_S + 1e-6 or too_long:
                segments.append(cur)
                cur = []
        cur.append(w)
    if cur:
        segments.append(cur)

    out = []
    for i, group in enumerate(segments):
        out.append({
            "id": i,
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": "".join(w["word"] for w in group).strip(),
            "avg_logprob": None,
            "no_speech_prob": None,
            "words": group,
        })
    return out


def build_record(rec, fid, src):
    words = normalise_words(rec.get("word_timestamps"))
    segments = to_segments(words)
    duration = max((w["end"] for w in words), default=0.0)
    return {
        "file_id": fid,
        "audio": str(src),
        "duration": round(duration, 3),
        "language": "en",
        "language_probability": None,
        "model": C.TRANSCRIPT_REPO,
        "beam_size": None,
        "source": "dataset_transcript",
        "n_words": len(words),
        "segments": segments,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jsonl = ensure_jsonl()
    manifest = {r["file_id"]: r for r in load_manifest()}
    ids = set(args.ids) if args.ids else None

    t0 = time.time()
    seen, written, skipped, empty = set(), 0, 0, 0
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            name = rec.get("filename") or ""
            fid = Path(name).stem
            if not fid:
                continue
            if ids is not None and fid not in ids:
                continue
            seen.add(fid)
            out = C.TRANSCRIPTS / f"{fid}.json"
            if out.exists() and not args.overwrite:
                skipped += 1
            else:
                words = rec.get("word_timestamps") or []
                if not words:
                    empty += 1
                    (C.TRANSCRIPTS / f"{fid}.empty.txt").write_text(
                        "no word timestamps in transcribe.jsonl\n", encoding="utf-8")
                else:
                    row = manifest.get(fid, {})
                    payload = build_record(rec, fid, row.get("path", ""))
                    tmp = out.with_suffix(".tmp")
                    tmp.write_text(json.dumps(payload, ensure_ascii=False),
                                   encoding="utf-8")
                    tmp.replace(out)
                    written += 1
                    if written % 100 == 0:
                        print(f"  ... {written} written, {skipped} already there "
                              f"({time.time() - t0:.0f}s)", flush=True)
            if args.limit and (written + skipped) >= args.limit:
                break

    missing = sorted(set(manifest) - seen) if not ids else []
    coverage = {
        "source": C.TRANSCRIPT_REPO + "/transcribe.jsonl",
        "records_seen": len(seen),
        "written": written,
        "already_present": skipped,
        "empty_transcripts": empty,
        "manifest_files": len(manifest),
        "manifest_missing": len(missing),
        "missing_examples": missing[:20],
    }
    (C.MANIFESTS / "transcript_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"written={written} present={skipped} empty={empty} "
          f"missing_from_corpus={len(missing)} in {(time.time() - t0) / 60:.1f} min")
    print("coverage:", C.MANIFESTS / "transcript_coverage.json")


if __name__ == "__main__":
    main()
