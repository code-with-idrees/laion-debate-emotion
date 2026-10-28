# LAION Debate Emotion

Speaker-turn level re-transcription, diarization and voice-model enrichment of
[`teamsleeping/sleeping-debate`](https://huggingface.co/datasets/teamsleeping/sleeping-debate)
(2,890 recordings, 436 GB, Oxford/Cambridge Union debates).

Everything lives under `D:\LAION_Debate_Emotion`.

## Pipeline

| # | script | what it does | output |
|---|--------|--------------|--------|
| 00 | `scripts/00_download_models.py` | ranged, resumable download of all model weights | `models/` |
| 01 | `scripts/01_download_dataset.py` | ranged, resumable download of the gated WAVs (`scripts\manifest_ok.py` counts progress) | `data/audio/`, `manifests/files.csv` |
| 02 | `scripts/02_diarize.py` | Nemotron 3 Diarization, offline chunked mode | `outputs/diarization/<id>.json` |
| 03 | `scripts/03_load_transcripts.py` | load the corpus' published word timestamps | `outputs/transcripts/<id>.json` |
| 04 | `scripts/04_build_turns.py` | word -> speaker alignment, turn building | `outputs/turns/<id>.jsonl` |
| 05 | `scripts/05_cut_snippets.py` | one 16 kHz mono FLAC per turn | `outputs/snippets/<id>/*.flac` |
| 06 | `scripts/06_score_empathic.py` | Empathic-Insight-Voice-Plus, 59 experts | `outputs/scores/<id>.empathic.jsonl` |
| 07 | `scripts/07_score_voicenet.py` | VoiceNet 57 regressions + 57 buckets | `outputs/scores/<id>.voicenet.jsonl` |
| 08 | `scripts/08_score_voiceclap.py` | VoiceCLAP-large-v2 3584-d embedding + top-5 | `outputs/scores/<id>.voiceclap.jsonl`, `outputs/embeddings/voiceclap/<id>.npy` |
| 09 | `scripts/09_assemble_dataset.py` | join everything | `dataset/data/train-00000-of-00001.parquet`, card, stats |

`run_pipeline.py` runs 01-09 in order (each stage in its own process so GPU
memory is released), and `status.py` prints per-stage progress.

## Quick start

```powershell
cd D:\LAION_Debate_Emotion
python scripts\00_download_models.py                 # ~52 GB, resumable
python scripts\01_download_dataset.py --workers 4     # needs gated access
python scripts\run_pipeline.py --from diarize
python scripts\status.py
```

Start every long-running job at once (download supervisor, live pipeline,
weights) — safe to re-run, it skips what is already alive:

```powershell
cd D:\LAION_Debate_Emotion
powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
```

Watch it live; Ctrl-C stops the display, not the download:

```powershell
cd D:\LAION_Debate_Emotion
python scripts\progress.py
```

`python scripts\progress.py --once` prints one snapshot (it samples until two
16 MB blocks have landed, so the speed is real; at most 60 s) and
`scripts\progress.cmd` does the `cd` for you and can be double-clicked. Everything is stored under `D:\LAION_Debate_Emotion`; the paths
come from the repository location, so another working directory changes nothing.

Smoke test on a subset:

```powershell
python scripts\run_pipeline.py --limit 5 --from diarize
```

Turn building, snippet cutting and assembly are CPU-bound and independent per
recording, so they use every core by default. Pass `--jobs N` (to the stage or
to `run_pipeline.py`) to reserve cores for something else.

## Design notes

- **Diarization**: Nemotron 3 native Transformers implementation
  (`nemotron_diarize.py`), offline 30.4 s chunks (27.2 s scored + 3.2 s right
  context) with a speaker cache; ~165x RTFx on an RTX 3090 once the first
  chunk has warmed the model (a 1.6 min file measures ~73x). Streaming modes are
  available via `02_diarize.py --mode`.
- **Transcripts**: the corpus ships its own word-level transcripts
  (`teamsleeping/sleeping-debate-transcribe/transcribe.jsonl`, 736 MB, 2,890
  records), so `03_load_transcripts.py` loads them instead of running ASR over
  436 GB a second time. Words are normalised to Whisper's shape (leading space,
  float timings) and grouped into 12 s / 0.4 s-pause segments, which is all
  `04_build_turns.py` needs. `03_transcribe.py` (Whisper large-v3 CT2,
  `vad_filter=True`, 40-60x RTFx) remains available via
  `run_pipeline.py --whisper-asr` for recordings the published file misses.
- **Turns**: 10 ms speaker activity matrix from diarization segments; each word
  goes to the speaker with the largest temporal overlap (midpoint tie-break);
  consecutive same-speaker words merge unless the speaker changes, the pause
  exceeds 0.8 s, or the turn exceeds 30 s.
- **Snippets**: FLAC `PCM_16` at 16 kHz mono, +/-50 ms padding - about half the
  disk of WAV.
- **Empathic**: all 59 experts (55 affect/attribute/pitch + 4 quality heads) run
  on one BUD-E-Whisper encoder pass; expert projections are fused into one fp16
  bank so 59 x 74M parameters fit in 24 GB.
- **VoiceNet**: one `voiceclap-commercial` embedding (768-d) feeds 57 regression
  and 57 classification heads. Bucket descriptions are long prose, so they are
  stored once in `dataset/voicenet_levels.json`, not per row.
- **VoiceCLAP**: 7B Qwen2.5-Omni thinker via `sentence-transformers` +
  `trust_remote_code`; embeddings normalised, stored fp16 in per-file `.npy`
  sidecars with a `.ids.json` order list; the table keeps top-5 text-probe
  similarities plus `voiceclap_emb_file`/`voiceclap_emb_row`.

## Resumability

Every stage is idempotent: an output file is skipped when it is complete and
valid (score files are checked against the snippet index segment ids). Partial
downloads resume from 16 MB block checkpoints (`*.blocks.json` next to the
`.part` file).
