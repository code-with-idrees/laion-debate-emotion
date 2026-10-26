"""Manual smoke test: offline-style 30.4 s chunking for Nemotron diarization.

Verifies the chunk arithmetic used by ``scripts/nemotron_diarize.py``:
27.2 s scored per chunk plus 3.2 s right context, with the speaker cache
carried across chunks.

Usage:  python tests/manual/smoke_chunked.py <wav>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import config as C  # noqa: E402

import librosa  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForAudioFrameClassification, AutoProcessor  # noqa: E402

wav_path = Path(sys.argv[1])

processor = AutoProcessor.from_pretrained(C.NEMOTRON_ID)
model = AutoModelForAudioFrameClassification.from_pretrained(C.NEMOTRON_ID).eval()
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

wav, _ = librosa.load(str(wav_path), sr=C.SR, mono=True)
dur = len(wav) / C.SR
print(f"audio {dur:.2f}s")

CHUNK_FRAMES, RIGHT_FRAMES = 340, 40           # 80 ms units
CHUNK_S = CHUNK_FRAMES * 0.08                  # 27.2 s scored
WIN_S = (CHUNK_FRAMES + RIGHT_FRAMES) * 0.08   # 30.4 s window


@torch.inference_mode()
def run():
    cache, outs = None, []
    t0 = time.time()
    start, n = 0.0, 0
    while start < dur - 0.01:
        end = min(start + WIN_S, dur)
        seg = wav[int(round(start * C.SR)):int(round(end * C.SR))]
        inputs = processor(seg, sampling_rate=C.SR)
        n_frames = inputs.input_features.shape[1] // 8
        lookahead = 0 if end >= dur - 1e-6 else min(RIGHT_FRAMES, max(0, n_frames - 1))
        inputs = inputs.to(model.device, dtype=model.dtype)
        out = model(**inputs, speaker_cache=cache, num_lookahead_frames=lookahead)
        outs.append(out.logits)
        cache = out.speaker_cache
        start += CHUNK_S
        n += 1
    logits = torch.cat(outs, dim=1)
    dt = time.time() - t0
    print(f"{int(n)} chunks, logits {tuple(logits.shape)}, {dt:.2f}s, RTFx {dur / dt:.1f}")
    segments = processor.extract_speaker_dict(logits)[0]
    print("segments", len(segments), segments[:5])
    return segments


run()
