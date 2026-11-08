# Build progress

A dated snapshot of the dataset build, so the numbers at any point can be read
without having to ask someone to run `status.py`. Regenerate the live numbers
with `python scripts\status.py` (or `scripts\stages.cmd`).

## 2026-11-03 23:30

### Corpus

| | |
|---|---|
| recordings in repo | 2,890 |
| downloaded | **2,890 / 2,890 (100%)** |
| bytes on disk | 436.43 / 436.42 GB |
| download errors | **0** |
| in-flight `.part` files | 0 |

The ranged downloader is checkpointed per 16 MB block, so a crash, a stall or a
power cut resumes instead of restarting. Throughput was limited by a 10 Mbps
Ethernet link (the adapter is a 2.5 GbE part that negotiated down), which is why
the pull took days rather than hours; see `runbook.md`.

### Derived audio

| stage | done | note |
|---|---|---|
| Nemotron 3 diarization | **2,890 / 2,890 (100%)** | 569–724× RTFx |
| corpus transcripts loaded | 2,877 / 2,890 (99.6%) | the corpus' own `transcribe.jsonl`; **no re-transcription** |
| speaker turns | 2,350 usable | 540 recordings have no usable transcript and are skip-marked |
| snippets cut | 2,350 files, **250,675 snippets** | 16 kHz mono FLAC, 62.7 GB |

2,350 + 540 = 2,890, so the audio side is complete: every recording is either
cut per speaker turn or deliberately skip-marked.

### Model scores

| model | scored | of 2,350 usable |
|---|---|---|
| VoiceNet (57 reg + 57 cls) | 1,459 | 62% |
| Empathic-Insight-Voice-Plus (55 emotion + 4 quality) | 918 | 39% |
| VoiceCLAP-large-v2 (3584-d embedding + top-5) | 918 | 39% |

Scoring runs on one RTX 3090 through `scripts/run_scores.ps1`, which chains the
three stages rather than starting them together (24 GB would not hold all
three), drains every snippet that exists, and loops as new ones are cut.

### Notes on data integrity

- `diarization_example.wav` is a test fixture. It leaked into the corpus manifest
  and was carried through diarisation, cutting and scoring; its row is now marked
  `example` so it is excluded from counts and from the assembled dataset.
- Two shutdowns interrupted the build. A FLAC truncated mid-write used to abort
  the whole scoring stage on every restart; a damaged snippet is now isolated,
  logged to `logs/bad_snippets.log`, and skipped.
- `scripts/start_all.ps1` restarts all four jobs after a reboot (corpus pull,
  live pipeline, weights, scoring) and is invoked by a Startup shortcut.

## 2026-11-05 22:30

The audio side is unchanged and complete (2,890 recordings, 250,675 snippets).
The two damaged snippets caused by the power cuts were re-cut and rescored.

| model | scored | pending of 2,335 recordings that have snippets |
|---|---|---|
| VoiceNet | 2,335 | **0** |
| Empathic-Insight | 2,335 | **0** |
| VoiceCLAP | 1,010 | **1,380** |

VoiceCLAP is the only stage left. It runs at 2.0 files/min at batch size 8
(measured over five minutes: 1.20 -> 2.00 files/min versus batch 2, with GPU
utilisation rising from 22% to 66%), so roughly 11 hours remain, after which
`09_assemble_dataset.py` runs automatically.

Fixes made while checking the endgame:

- The pending check never actually ran: PowerShell 5.1 strips double quotes from
  a multi-line `python -c` here-string, so it printed nothing and the loop ran
  every stage on every pass. It is now `scripts/pending.py`, covered by tests.
- Recordings that align to zero turns have an empty snippet index and can never
  be scored; they were counted as pending forever, which would have stopped
  assembly from ever firing. Pending now requires at least one snippet.
- A failed snippet could shift VoiceCLAP's embedding row indices, pointing
  later turns at the wrong voice vector. The `.npy` writer records `ids.json`
  from the actual pairs and assembly resolves through it.
- `scripts/verify_dataset.py` checks the final Parquet's invariants, including
  every embedding pointer, so the dataset can be signed off in one command.

85 tests pass.

## 2026-11-08 18:43 - complete

The build finished and the dataset verified clean.

| | |
|---|---|
| rows | **250,620 speaker turns** |
| source recordings | 2,335 |
| speakers | 8 |
| speech | **963.19 hours** |
| coverage | empathic **100%**, voicenet **100%**, voiceclap **100%** |
| parquet | `dataset/data/train-00000-of-00001.parquet` (0.34 GB) |
| dropped | 32 zero-length turns |

`scripts/verify_dataset.py` reports `OK - no invariant violations` over all
250,620 rows, including every VoiceCLAP embedding pointer.

Not in the dataset by design: 555 recordings - 540 whose published transcript is
empty or missing, and 15 that align to zero turns. Recovering them needs the
opt-in Whisper fallback (`run_pipeline.py --whisper-asr`), not the corpus
transcripts.

Verified failure modes found before assembly, all fixed:

- a stale pending count that would have looped forever without assembling;
- an unreadable snippet that aborted the whole stage;
- embedding row indices that shifted by one after a failed snippet;
- turns whose closing word carried a zero timestamp.


