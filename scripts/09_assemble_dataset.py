"""09 - Assemble the enriched dataset.

Joins, per speaker turn:
  outputs/turns/<fid>.jsonl              (diarization + ASR aligned turns)
  outputs/scores/<fid>.empathic.jsonl    (59 Empathic-Insight-Voice-Plus experts)
  outputs/scores/<fid>.voicenet.jsonl    (57 VoiceNet regressions + 57 buckets)
  outputs/scores/<fid>.voiceclap.jsonl   (top-5 text probes, 3584-d embedding)

into one Parquet table (one row group per source file) plus a dataset card,
stats, and a small JSONL sample.  Full VoiceCLAP embeddings stay in
outputs/embeddings/voiceclap/<fid>.npy and are referenced by file + row index.

Usage:
  python 09_assemble_dataset.py
  python 09_assemble_dataset.py --limit 20 --out dataset_debug.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import config as C
import score_common as S

EMB_REL = "outputs/embeddings/voiceclap"


def read_jsonl(path: Path):
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                out[r["segment_id"]] = r
    return out


def f32(v):
    return None if v is None else np.float32(v)


def build_record(fid, turn, emp, vn, vc, emb_file, emb_row):
    v = (vn or {}).get("voicenet")
    if v:
        v = {
            "reg": {k: f32(x) for k, x in (v.get("reg") or {}).items()},
            "cls": {k: np.int8(x) for k, x in (v.get("cls") or {}).items()},
            "conf": {k: f32(x) for k, x in (v.get("conf") or {}).items()},
        }
    e = (emp or {}).get("empathic")
    if e:
        e = {k: f32(x) for k, x in e.items()}
    e_top = (emp or {}).get("empathic_top5")
    if e_top:
        e_top = [{"label": t["label"], "score": f32(t["score"])} for t in e_top]
    c_top = (vc or {}).get("voiceclap_top5")
    if c_top:
        c_top = [{"label": t["label"], "sim": f32(t["sim"])} for t in c_top]
    return {
        "file_id": fid,
        "segment_id": turn["segment_id"],
        "start": float(turn["start"]),
        "end": float(turn["end"]),
        "duration": float(turn.get("duration") or (turn["end"] - turn["start"])),
        "speaker": turn.get("speaker"),
        "text": turn.get("text"),
        "n_words": np.int32(turn.get("n_words") or 0),
        "overlap": f32(turn.get("overlap") or 0.0),
        "asr_confidence": f32(turn.get("asr_confidence")),
        "empathic": e,
        "empathic_top5": e_top,
        "voicenet": v,
        "voiceclap_top5": c_top,
        "voiceclap_best": (vc or {}).get("voiceclap_best"),
        "voiceclap_best_sim": f32((vc or {}).get("voiceclap_best_sim")),
        "voiceclap_emb_file": emb_file,
        "voiceclap_emb_row": (None if emb_row is None else np.int32(emb_row)),
    }


def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def collect_rows(fid, stats):
    turns_p = C.TURNS / f"{fid}.jsonl"
    if not turns_p.exists():
        return None, "no_turns"
    turns = [json.loads(l) for l in turns_p.open(encoding="utf-8") if l.strip()]
    emp = read_jsonl(S.out_path("empathic", fid))
    vn = read_jsonl(S.out_path("voicenet", fid))
    vc = read_jsonl(S.out_path("voiceclap", fid))

    emb_file = f"{EMB_REL}/{fid}.npy" if (C.EMBEDDINGS / "voiceclap" / f"{fid}.npy").exists() else None
    rows, emb_row = [], 0
    for t in turns:
        sid = t["segment_id"]
        e, n, c = emp.get(sid), vn.get(sid), vc.get(sid)
        if c is not None:
            row_idx = emb_row
        else:
            row_idx = None
        rows.append(build_record(fid, t, e, n, c, emb_file, row_idx))
        if c is not None:
            emb_row += 1

    stats["files"] += 1
    stats["turns"] += len(rows)
    stats["speech_s"] += sum(r["duration"] for r in rows)
    stats["with_empathic"] += sum(1 for r in rows if r["empathic"] is not None)
    stats["with_voicenet"] += sum(1 for r in rows if r["voicenet"] is not None)
    stats["with_voiceclap"] += sum(1 for r in rows if r["voiceclap_top5"] is not None)
    for r in rows:
        if r["empathic_top5"]:
            stats["empathic_top1"][r["empathic_top5"][0]["label"]] += 1
        if r["voiceclap_best"]:
            stats["voiceclap_top1"][r["voiceclap_best"]] += 1
        stats["speakers"].add(r["speaker"])
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample-rows", type=int, default=500)
    args = ap.parse_args()

    manifest = S.load_manifest(args.ids, args.limit)
    print(f"files: {len(manifest)}")

    out = Path(args.out) if args.out else C.DATASET / "data" / "train-00000-of-00001.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")

    stats = {"files": 0, "turns": 0, "speech_s": 0.0, "with_empathic": 0,
             "with_voicenet": 0, "with_voiceclap": 0,
             "empathic_top1": Counter(), "voiceclap_top1": Counter(),
             "speakers": set()}
    stats["speakers"] = set()

    schema = None
    skipped = []
    sample = []
    deferred = []
    t0 = time.time()
    writer = None
    try:
        # Pass 1: fully-scored files establish the schema.
        order = sorted(manifest, key=lambda r: r["file_id"])
        for i, m in enumerate(order, 1):
            fid = m["file_id"]
            rows, err = collect_rows(fid, stats)
            if err:
                skipped.append({"file_id": fid, "reason": err})
                continue
            if not rows:
                skipped.append({"file_id": fid, "reason": "empty"})
                continue
            if schema is None:
                if (rows[0]["empathic"] is None or rows[0]["voicenet"] is None
                        or rows[0]["voiceclap_top5"] is None):
                    deferred.append((fid, rows))
                    continue
                table = pa.Table.from_pylist(rows)
                schema = table.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            else:
                try:
                    table = pa.Table.from_pylist(rows, schema=schema)
                except Exception as e:
                    skipped.append({"file_id": fid, "reason": f"schema:{e}"})
                    continue
            writer.write_table(table, row_group_size=max(1, len(rows)))
            if len(sample) < args.sample_rows:
                sample.extend(rows[: max(0, args.sample_rows - len(sample))])
            if i % 25 == 0 or i == len(order):
                print(f"[{i}/{len(order)}] {fid}: {len(rows)} rows "
                      f"({time.time()-t0:.0f}s)", flush=True)

        # Pass 2: files missing one scorer, written as nulls against the schema.
        for fid, rows in deferred:
            try:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema),
                                   row_group_size=max(1, len(rows)))
                skipped.append({"file_id": fid, "reason": "written_partial"})
            except Exception as e:
                skipped.append({"file_id": fid, "reason": f"schema:{e}"})
    finally:
        if writer is not None:
            writer.close()

    if schema is None:
        print("nothing assembled; no file had complete scores")
        return

    tmp.replace(out)

    # sample + card + stats
    with (C.DATASET / "sample.jsonl").open("w", encoding="utf-8") as fh:
        for r in sample:
            fh.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

    (C.DATASET / "schema.txt").write_text(str(schema), encoding="utf-8")

    stats_out = {
        "dataset": C.DATASET_REPO,
        "source_audio": C.DATASET_REPO,
        "rows": stats["turns"],
        "files": stats["files"],
        "speakers": len(stats["speakers"]),
        "speech_hours": round(stats["speech_s"] / 3600, 3),
        "coverage": {
            "empathic": round(stats["with_empathic"] / max(stats["turns"], 1), 4),
            "voicenet": round(stats["with_voicenet"] / max(stats["turns"], 1), 4),
            "voiceclap": round(stats["with_voiceclap"] / max(stats["turns"], 1), 4),
        },
        "empathic_top1": stats["empathic_top1"].most_common(20),
        "voiceclap_top1": stats["voiceclap_top1"].most_common(20),
        "skipped": skipped,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parquet": str(out),
    }
    (C.DATASET / "stats.json").write_text(json.dumps(stats_out, indent=2),
                                          encoding="utf-8")
    write_card(stats_out, schema)
    print(json.dumps({k: v for k, v in stats_out.items()
                      if k not in ("skipped", "empathic_top1", "voiceclap_top1")},
                     indent=2))
    print(f"wrote {out} ({out.stat().st_size/1e9:.2f} GB)")
    if skipped:
        print(f"skipped {len(skipped)} files; see dataset/stats.json")


def write_card(stats, schema):
    cols = "\n".join(
        f"| `{f.name}` | `{f.type}` |" for f in schema)
    card = f"""---
license: other
task_categories:
- audio-classification
- text-classification
language:
- en
tags:
- audio
- speech
- emotion
- diarization
- debate
- laion
pretty_name: Sleeping Debate - Speaker-Turn Emotion & Voice Attributes
size_categories:
- 100K<n<1M
---

# Sleeping Debate: speaker-turn emotion & voice-attribute enrichment

Re-transcription, re-diarization and per-turn voice-model scoring of
[`{stats['dataset']}`](https://huggingface.co/datasets/{stats['dataset']}).

## Pipeline

1. **Diarization** - NVIDIA Nemotron 3 Diarization (`nvidia/Nemotron-3-Diarization`),
   offline chunked mode (27.2 s scored + 3.2 s right context, speaker cache).
2. **Transcription** - Whisper `large-v3` via CTranslate2, word-level timestamps.
3. **Turns** - words aligned to diarization speakers by temporal overlap;
   consecutive same-speaker words are merged (split on speaker change,
   pause > 0.8 s, or turn > 30 s).
4. **Snippets** - one 16 kHz mono FLAC clip per turn (`outputs/snippets/`).
5. **Models**, each run on every turn clip separately:
   - `laion/voiceclap-large-v2` -> 3584-d audio embedding + top-5 text probes
   - `laion/Empathic-Insight-Voice-Plus` -> 55 experts + 4 audio-quality scores
   - `laion/voicenet-dimension-predictors-commercial` -> 57 regressions +
     57 classification buckets with confidences

## Statistics

| | |
|---|---|
| rows (speaker turns) | {stats['rows']:,} |
| source files | {stats['files']:,} |
| speech hours | {stats['speech_hours']:,} |
| coverage | empathic {stats['coverage']['empathic']:.1%}, voicenet {stats['coverage']['voicenet']:.1%}, voiceclap {stats['coverage']['voiceclap']:.1%} |

## Columns

| column | type |
|---|---|
{cols}

### Notes

- `empathic` holds 59 named expert scores (55 affect/attribute/pitch experts +
  4 `*_Quality`/`Content_Enjoyment` heads). Values are the raw expert outputs.
- `empathic_top5` / `voiceclap_top5` are compact convenience views: the five
  highest-scoring labels with their scores.
- `voicenet.reg` are calibrated regression scores per dimension;
  `voicenet.cls` the predicted bucket id and `voicenet.conf` its probability.
  Bucket descriptions are long prose, so they live once in
  `voicenet_levels.json` (`dim -> levels[bucket_id]`) instead of being repeated
  on every row.
- `voiceclap_emb_file` + `voiceclap_emb_row` point into the fp16 embedding
  store (`outputs/embeddings/voiceclap/<file_id>.npy`, one row per turn in
  `segment_id` order). Load with `np.load(...)[row]`.
- Timestamps are seconds within the source recording; `speaker` is the
  diarization label local to that recording.

Built {stats['built_at']}.
"""
    (C.DATASET / "README.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
