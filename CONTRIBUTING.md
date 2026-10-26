# Contributing

Thanks for taking a look. This is a research pipeline, but the code is meant to
be readable and re-runnable on a fresh machine.

## Layout

```
scripts/          numbered pipeline stages (01-09) + shared modules
  config.py       every path, model id and constant
  hf_fetch.py     resumable ranged downloader
  run_pipeline.py stage driver; status.py progress report
tests/            offline unit tests (no GPU, no network)
tests/manual/     GPU smoke tests against a real WAV
examples/         downstream helpers (dataset inspector)
docs/models.md    what each model does and which output shape we rely on
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Data, weights and outputs default to the repository root. To keep them on a
different disk, set `LAION_DEBATE_ROOT` before running anything:

```powershell
$env:LAION_DEBATE_ROOT = "D:\LAION_Debate_Emotion"
```

CPU-side unit tests need no token and no downloads:

```powershell
python -m unittest discover -s tests -v
```

The GPU smoke tests need the model weights in `models/` and a sample WAV; see
`tests/manual/README.md`.

## Adding or changing a stage

1. Read `scripts/score_common.py`. Scoring stages are file-oriented: read the
   snippet index, write one atomic JSONL, and make the output complete-able by
   re-running the same command.
2. Never hard-code a path. Add it to `config.py` and import config.
3. Keep row payloads compact. Large arrays belong in a sidecar (`.npy`) with a
   file/row reference in the table, and repeated prose belongs in a JSON
   sidecar (`dataset/voicenet_levels.json` is the precedent).
4. Add a stage entry to `run_pipeline.py` (`STAGES`) and `status.py` so the
   orchestration and the progress report stay truthful.
5. Add offline tests for any pure logic (alignment, batching, checkpointing).
   If the change is model-side only, extend the relevant manual smoke test.
6. Update the README pipeline table and `docs/models.md` when the output
   changes.

## Style

- Python 3.11+, `from __future__ import annotations`, type hints on public
  functions.
- `pathlib.Path` for paths; no `os.path` string juggling.
- Log progress with plain `print(..., flush=True)`; the pipeline is run under
  `run_pipeline.py` and logs are captured to files.
- Fail loudly at the file level (write `<id>.error.txt` and continue) so a
  multi-day run is not lost to one bad recording.

## Commits

One logical change per commit, conventional-commit subjects
(`feat:`, `fix:`, `docs:`, `test:`, `chore:`) and a body that explains *why*.
Code, tests, docs and examples are committed separately so the history shows
what was added and what was verified.

## Data

`teamsleeping/sleeping-debate` is a gated dataset: do not commit audio, model
weights, embeddings or the assembled Parquet. `.gitignore` already excludes
`data/`, `models/`, `outputs/`, `dataset/`, `cache/`, `logs/` and `manifests/`.
Anyone re-running the pipeline must request access on the Hub themselves.
