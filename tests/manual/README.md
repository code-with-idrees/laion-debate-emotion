# Manual smoke tests

These are **not** part of the automated suite: they load the real models onto
the GPU and need a sample WAV plus the downloaded weights in `models/`.

```powershell
python tests\manual\smoke_diarize.py data\audio\diarization_example.wav
python tests\manual\smoke_chunked.py data\audio\diarization_example.wav
python tests\manual\smoke_whisper.py  data\audio\diarization_example.wav
```

Each script prints load time, throughput (RTFx) and the first few decoded
segments, which is enough to confirm that a fresh environment is healthy before
starting the multi-day run.
