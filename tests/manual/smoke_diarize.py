"""Manual smoke test: Nemotron 3 Diarization via Transformers (single pass).

Usage:  python tests/manual/smoke_diarize.py <wav>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import config as C  # noqa: E402  (sets HF_HOME, HF_TOKEN, CUDA DLL dirs)

import librosa  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForAudioFrameClassification, AutoProcessor  # noqa: E402

wav_path = Path(sys.argv[1])
t0 = time.time()
processor = AutoProcessor.from_pretrained(C.NEMOTRON_ID)
model = AutoModelForAudioFrameClassification.from_pretrained(C.NEMOTRON_ID)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
print(f"loaded in {time.time() - t0:.1f}s device={model.device} dtype={model.dtype}")

print("processor methods:",
      [m for m in dir(processor) if "chunk" in m or "speaker" in m or "stream" in m])
for attr in ("num_samples_first_audio_chunk", "num_samples_per_audio_chunk",
             "num_mel_frames_per_step"):
    print(f"{attr}:", getattr(processor, attr, None))

wav, sr = librosa.load(str(wav_path), sr=C.SR, mono=True)
print("audio", wav.shape, sr, f"{len(wav) / sr:.2f}s")

t1 = time.time()
inputs = processor(wav, sampling_rate=sr).to(model.device, dtype=model.dtype)
with torch.inference_mode():
    logits = model(**inputs).logits
print("logits", tuple(logits.shape), f"in {time.time() - t1:.2f}s")
segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
print("segments:", segments[:10], "... total", len(segments))
