"""Manual smoke test: faster-whisper large-v3 (CT2 float16) with word timestamps.

Usage:  python tests/manual/smoke_whisper.py <wav>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import config as C  # noqa: E402  (CUDA DLL paths)

from faster_whisper import BatchedInferencePipeline, WhisperModel  # noqa: E402

t0 = time.time()
model = WhisperModel(str(C.WHISPER_CT2), device="cuda", compute_type="float16",
                     cpu_threads=4)
print(f"loaded {time.time() - t0:.1f}s")
pipe = BatchedInferencePipeline(model=model)

t1 = time.time()
segments, info = pipe.transcribe(
    sys.argv[1], batch_size=16, language="en", task="transcribe",
    word_timestamps=True, vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=300),
    condition_on_previous_text=False, beam_size=1,
)
segs = list(segments)
dt = time.time() - t1
print(f"lang={info.language} p={info.language_probability:.2f} "
      f"dur={info.duration:.1f}s time={dt:.1f}s RTFx={info.duration / dt:.0f} "
      f"segments={len(segs)}")
for s in segs[:3]:
    print(f"[{s.start:.2f}-{s.end:.2f}] {s.text}")
    print("  words:", [(w.word, round(w.start, 2), round(w.end, 2))
                       for w in (s.words or [])][:6])
