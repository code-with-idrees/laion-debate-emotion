# Runbook

Operational notes for the full corpus (2,890 recordings, 436 GB, ~1,375 hours).
Everything here was measured on one RTX 3090 (24 GB) on Windows.

## Before the first run

```powershell
python scripts\preflight.py        # exits non-zero if the run cannot start
```

`preflight.py` checks the GPU stack, free disk against the projected 636 GB,
per-group weight completeness and whether the current token may actually read
the gated dataset (a 1-byte Range request; listing metadata is not enough).

Dataset access is granted manually on the Hub. Until it is, stage 01 stops with
`ACCESS DENIED` and the rest of the pipeline has nothing to do.

## Order of work

Weights first, then the corpus:

```powershell
python scripts\00_download_models.py --only voicenet          # 0.5 GB, minutes
python scripts\00_download_models.py --only bude              # 1.0 GB, minutes
python scripts\00_download_models.py --only voiceclap         # 18 GB, hours
python scripts\00_download_models.py --only empathic          # 16 GB, hours
python scripts\01_download_dataset.py --workers 4             # 436 GB, ~1-3 days
python scripts\run_pipeline.py --from diarize                 # ~2 days of GPU
```

Downloads are independent of GPU work, so they can overlap with scoring. Do not
run two GPU stages at once: each model wants most of the 24 GB.

Turn building, snippet cutting and assembly fan out over every logical core by
default (`--jobs N` to cap it, e.g. `--jobs 2` while a GPU stage is running so
data loading still gets CPU).

## Measured throughput

| stage | measured | full corpus |
|-------|----------|-------------|
| diarization | 620x RTFx | ~2.2 h |
| ASR (large-v3, batch 16) | 36-39x RTFx | ~36 h |
| turn building | CPU, all cores | ~1 h |
| snippet cutting | CPU, all cores (FLAC encode) | ~5 h |
| Empathic (59 experts) | one encoder pass per batch | ~10-20 h |
| VoiceNet | encoder + 114 heads | ~5-10 h |
| VoiceCLAP (7B) | 3-6 clips/s | 3-5 days |
| assembly | CPU, all cores | ~30 min |

VoiceCLAP dominates. If time matters more than full embeddings, `--quant 8bit`
fits more headroom for a larger batch; the embedding stays 3584-d.

## Monitoring

```powershell
python scripts\status.py                        # per-stage counts
python scripts\preflight.py                     # stack/disk/weights/access
Get-Content logs\dl_weights.log -Tail 20        # download rate
Get-Content logs\pipeline.log -Tail 50          # stage output
```

`run_pipeline.py` prints to the console; start it with a redirection if you want
a file to tail:

```powershell
python scripts\run_pipeline.py --from diarize *> logs\pipeline.log
```

Every stage is idempotent: re-running the same command skips files whose output
already exists and is complete. `--overwrite` forces a redo of one stage.

## Failure modes

- **Gated 403** — request access on the dataset page; stage 01 exits with
  `ACCESS DENIED` rather than retrying.
- **Download stall** — each block is an independent Range request with a 30 s
  read timeout; stalled blocks retry, completed blocks are checkpointed in
  `*.blocks.json`. Restarting the same command resumes. Changing `--block-mb`
  discards the checkpoint (the partial file stays valid).
- **CUDA OOM** — the scoring harness halves the batch and retries down to a
  single snippet; persistent OOM means lower `--batch-size`.
- **Bad recording** — a file-level exception writes
  `outputs/<stage>/<id>.error.txt` and the run continues. Inspect those files
  before assembling: `09_assemble_dataset.py` only joins complete scores.
- **Disk pressure** — largest consumers are the raw WAVs (436 GB), snippets
  (~150 GB) and VoiceCLAP embeddings (~10 GB). Snippets can be deleted once all
  four score files for a recording exist; raw audio only after every stage has
  consumed it.

## Restarting after a crash

```powershell
python scripts\status.py                 # find the first incomplete stage
python scripts\run_pipeline.py --from snippets
```

Individual stages accept `--ids <file_id>` (or `--limit N`) for debugging, e.g.

```powershell
python scripts\08_score_voiceclap.py --ids diarization_example
```

`tests/manual/` has the three GPU smoke tests used to verify a machine before a
multi-day run.
