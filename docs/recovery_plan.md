# Recovery plan for the 28 recordings still excluded

After the 527 stale-skip recoveries, 28 recordings remain outside the
dataset. They fall into two buckets:

- `logs/_need_diarize.txt` (15): diarization produced no usable segments, or
  alignment produced zero turns. Re-run `02_diarize.py --overwrite` on them;
  recordings that still yield zero segments are genuinely speechless (or
  un-diarizable) and stay excluded.
- `logs/_need_whisper.txt` (13): no usable transcript words. Run the opt-in
  fallback `run_pipeline.py --whisper-asr --stages transcribe turns snippets
  --ids <id>`.

Once any of them produce snippets, score with the three models (voicenet,
empathic, voiceclap --batch-size 8) and re-run `09_assemble_dataset.py`,
then `verify_dataset.py`.

Execution order (GPU is single-card, so strictly sequential):

1. `python scripts\02_diarize.py --ids (cat logs/_need_diarize.txt) --overwrite`
2. `python scripts\run_pipeline.py --whisper-asr --stages transcribe turns snippets --ids (cat logs/_need_whisper.txt)`
3. `python scripts\04_build_turns.py --ids (cat logs/_need_diarize.txt) --overwrite`
4. `python scripts\05_cut_snippets.py --ids <newly turned> --overwrite`
5. Score the newly snippet-bearing ids with all three models.
6. `python scripts\09_assemble_dataset.py` then `python scripts\verify_dataset.py`.

VoiceCLAP on the 527 recovered recordings is still running (started ~17:07).
