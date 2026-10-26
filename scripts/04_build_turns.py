"""04 - Align ASR words to diarization speakers and build speaker turns.

Inputs : outputs/diarization/<id>.json
         outputs/transcripts/<id>.json
Output : outputs/turns/<id>.jsonl   (one row per speaker turn)
         outputs/turns/<id>.stats.json

Alignment
---------
A 10 ms speaker-activity matrix is built from the diarization segments.  Every
Whisper word is assigned to the speaker with the largest temporal overlap; ties
are broken by the speaker active at the word midpoint.  Consecutive words with
the same speaker are grouped into a turn, splitting on speaker changes, pauses
> 0.8 s, or turns longer than 30 s.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from parallel import run_rows


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


def build_activity(segments, duration):
    """Return (speaker_ids, activity[n_spk, n_frames], n_frames)."""
    segs = []
    for s in segments:
        a, b = float(s["Start"]), float(s["End"])
        if b <= a:
            continue
        segs.append((a, b, int(s["Speaker"])))
    speaker_ids = sorted({s[2] for s in segs})
    idx = {spk: i for i, spk in enumerate(speaker_ids)}
    n_frames = max(1, int(math.ceil(duration / C.FRAME_MS * 1000)))
    act = np.zeros((len(speaker_ids), n_frames), dtype=bool)
    for a, b, spk in segs:
        f0 = max(0, int(a * 1000 / C.FRAME_MS))
        f1 = min(n_frames, int(math.ceil(b * 1000 / C.FRAME_MS)))
        if f1 > f0:
            act[idx[spk], f0:f1] = True
    return speaker_ids, act, n_frames


def word_speaker(w, act, speaker_ids):
    """Return (speaker_id or None, overlap_fraction)."""
    if not speaker_ids:
        return None, 0.0
    n_frames = act.shape[1]
    a = w.get("start")
    b = w.get("end")
    if a is None and b is None:
        return None, 0.0
    if a is None:
        a = b
    if b is None:
        b = a
    f0 = max(0, min(n_frames - 1, int(a * 1000 / C.FRAME_MS)))
    f1 = max(f0 + 1, min(n_frames, int(math.ceil(b * 1000 / C.FRAME_MS))))
    counts = act[:, f0:f1].sum(axis=1)
    if counts.max() == 0:
        # nearest active frame within +/- 500 ms
        lo, hi = max(0, f0 - 50), min(n_frames, f1 + 50)
        near = act[:, lo:hi].sum(axis=1)
        if near.max() == 0:
            return None, 0.0
        counts = near
    spk_idx = int(np.argmax(counts))
    if counts[spk_idx] == 0:
        return None, 0.0
    # tie-break with midpoint activity
    tied = np.flatnonzero(counts == counts[spk_idx])
    if len(tied) > 1:
        mid = max(0, min(n_frames - 1, int(((a + b) / 2) * 1000 / C.FRAME_MS)))
        active_mid = [i for i in tied if act[i, mid]]
        if active_mid:
            spk_idx = active_mid[0]
    overlap = float((act[:, f0:f1].sum(axis=0) > 1).mean())
    return speaker_ids[spk_idx], overlap


def flush(group, fid, idx, out):
    if not group:
        return idx
    text = "".join(w["word"] for w in group).strip()
    if not text:
        return idx
    start = min(w["start"] for w in group)
    end = max(w["end"] for w in group)
    if end - start < C.TURN_MIN_S and len(group) < 2:
        return idx
    probs = [w["probability"] for w in group if w.get("probability") is not None]
    row = {
        "segment_id": f"{fid}_{idx:05d}",
        "file_id": fid,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "speaker": f"speaker_{group[0]['speaker']}",
        "text": text,
        "n_words": sum(1 for w in group if w["word"].strip()),
        "overlap": round(float(np.mean([w["overlap"] for w in group])), 3),
        "asr_confidence": round(float(np.mean(probs)), 4) if probs else None,
        "words": [{"word": w["word"], "start": w["start"], "end": w["end"],
                   "probability": w["probability"]} for w in group],
    }
    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return idx + 1


def group_words(words):
    """Yield consecutive word groups that form one speaker turn.

    A new group starts on a speaker change, a pause longer than
    ``TURN_GAP_S``, or when the group would exceed ``TURN_MAX_S``.
    """
    group = []
    for w in words:
        if w.get("speaker") is None:
            continue
        if group:
            gap = w["start"] - group[-1]["end"]
            too_long = (w["end"] - group[0]["start"]) > C.TURN_MAX_S
            if w["speaker"] != group[-1]["speaker"] or gap > C.TURN_GAP_S or too_long:
                yield group
                group = []
        group.append(w)
    if group:
        yield group


def process(fid, overwrite=False):
    dpath = C.DIAR / f"{fid}.json"
    tpath = C.TRANSCRIPTS / f"{fid}.json"
    out_path = C.TURNS / f"{fid}.jsonl"
    if out_path.exists() and not overwrite:
        return None
    if not dpath.exists() or not tpath.exists():
        return {"file_id": fid, "status": "missing_input"}

    diar = json.loads(dpath.read_text(encoding="utf-8"))
    tr = json.loads(tpath.read_text(encoding="utf-8"))
    duration = float(diar.get("duration") or tr.get("duration") or 0.0)
    speaker_ids, act, n_frames = build_activity(diar.get("segments", []), duration)

    # Flatten word list, keep only words with timings.
    words = []
    for seg in tr.get("segments", []):
        for w in seg.get("words") or []:
            if not w.get("word") or not w["word"].strip():
                continue
            a, b = w.get("start"), w.get("end")
            if a is None and b is None:
                continue
            if a is None:
                a = b
            if b is None:
                b = a
            if b < a:
                a, b = b, a
            words.append({"word": w["word"], "start": float(a), "end": float(b),
                          "probability": w.get("probability")})
    words.sort(key=lambda x: (x["start"], x["end"]))

    # Assign speakers.
    for w in words:
        spk, ov = word_speaker(w, act, speaker_ids)
        w["speaker"] = spk
        w["overlap"] = ov

    # Fill unassigned words with the previous/next assigned speaker.
    last = None
    for w in words:
        if w["speaker"] is None:
            w["speaker"] = last
            w["overlap"] = 0.0
        else:
            last = w["speaker"]
    nxt = None
    for w in reversed(words):
        if w["speaker"] is None:
            w["speaker"] = nxt
        else:
            nxt = w["speaker"]

    tmp = out_path.with_suffix(".tmp")
    idx = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for group in group_words(words):
            idx = flush(group, fid, idx, fh)
    tmp.replace(out_path)

    # Stats
    rows = [json.loads(line) for line in out_path.open(encoding="utf-8")]
    talk = defaultdict(float)
    for r in rows:
        talk[r["speaker"]] += r["duration"]
    speech_s = sum(r["duration"] for r in rows)
    stats = {
        "file_id": fid,
        "duration": round(duration, 3),
        "n_turns": len(rows),
        "n_words": sum(r["n_words"] for r in rows),
        "speech_s": round(speech_s, 2),
        "speech_fraction": round(speech_s / duration, 4) if duration else None,
        "overlap_fraction": round(speech_s and sum(r["duration"] * r["overlap"] for r in rows) / speech_s, 4),
        "n_speakers": len(speaker_ids),
        "speaker_talk_s": {str(k): round(v, 2) for k, v in sorted(talk.items())},
        "asr_confidence_mean": round(float(np.mean([r["asr_confidence"] for r in rows
                                                    if r["asr_confidence"] is not None])), 4)
        if any(r["asr_confidence"] is not None for r in rows) else None,
    }
    (C.TURNS / f"{fid}.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def process_row(row, overwrite=False):
    """Manifest row -> stats dict (worker entry point for the process pool)."""
    return process(row["file_id"], overwrite)


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
    ok = 0
    for i, total, row, status, payload in run_rows(
            rows, process_row, args.jobs, args.overwrite):
        fid = row["file_id"]
        if status == "error":
            print(f"FAIL {fid}", flush=True)
            print(payload, flush=True)
            (C.TURNS / f"{fid}.error.txt").write_text(payload, encoding="utf-8")
            continue
        st = payload
        if st:
            ok += 1
            print(f"[{i}/{total}] {fid} -> {st.get('n_turns', st.get('status'))} turns"
                  + (f", {st['n_speakers']} speakers, "
                     f"overlap {st['overlap_fraction']}" if "n_speakers" in st else ""),
                  flush=True)
    print(f"done {ok}")


if __name__ == "__main__":
    main()
