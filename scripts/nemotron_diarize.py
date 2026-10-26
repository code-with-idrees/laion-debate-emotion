"""Nemotron 3 Diarization (Transformers) - long-audio inference.

Two modes:
  * ``offline`` (default): 30.4 s windows (27.2 s scored + 3.2 s right context)
    driven by the model's own speaker cache.  ~300x realtime on an RTX 3090.
  * ``low_latency``: the processor's 1.04 s streaming chunks (slower, for
    compatibility / debugging).

Output JSON:
  {"audio":..., "duration":..., "model":..., "mode":..., "rtfx":...,
   "segments":[{"Start":0.4,"End":2.1,"Speaker":0}, ...]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

sys.path.insert(0, str(Path(__file__).parent))
import config as C  # noqa: E402  (sets HF_HOME / HF_TOKEN / CUDA DLL paths)

MODEL_ID = C.NEMOTRON_ID
SR = C.SR
FRAME_S = 0.08          # encoder frame (80 ms)
CHUNK_FRAMES = 340      # scored frames per window (27.2 s)
RIGHT_FRAMES = 40       # right context (3.2 s)


def load_audio_16k(path: str | Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def load_model(device: str = "cuda"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        MODEL_ID, dtype=torch.float32
    )
    model.eval().to(device)
    return processor, model


@torch.inference_mode()
def diarize_chunked(processor, model, wav: np.ndarray, device: str = "cuda"):
    """Offline-style 30.4 s windows with the model speaker cache."""
    cache = None
    logits_parts = []
    total_frames = 0
    start_frame = 0
    dur = len(wav) / SR
    n_chunks = 0
    while start_frame * FRAME_S < dur - 1e-4:
        start = start_frame * FRAME_S
        is_last = (start + (CHUNK_FRAMES + RIGHT_FRAMES) * FRAME_S) >= dur - 1e-4
        end = dur if is_last else start + (CHUNK_FRAMES + RIGHT_FRAMES) * FRAME_S
        seg = wav[int(round(start * SR)):int(round(end * SR))]
        if len(seg) < 1600:  # < 0.1 s
            break
        inputs = processor(seg, sampling_rate=SR)
        n_enc = inputs.input_features.shape[1] // 8
        lookahead = 0 if is_last else min(RIGHT_FRAMES, max(0, n_enc - 1))
        inputs = inputs.to(device, dtype=model.dtype)
        out = model(**inputs, speaker_cache=cache, num_lookahead_frames=lookahead)
        logits_parts.append(out.logits)
        cache = out.speaker_cache
        scored_enc = n_enc - lookahead
        start_frame += scored_enc
        total_frames += out.logits.shape[1]
        n_chunks += 1
    if not logits_parts:
        return [], None, 0
    logits = torch.cat(logits_parts, dim=1)
    segments = processor.extract_speaker_dict(logits)[0]
    return segments, logits, n_chunks


@torch.inference_mode()
def diarize_streaming(processor, model, wav: np.ndarray, mode: str = "low_latency",
                      device: str = "cuda"):
    processor.set_streaming_mode(mode)
    audio = torch.as_tensor(wav, dtype=torch.float32)
    first = processor.num_samples_first_audio_chunk
    per = processor.num_samples_per_audio_chunk
    mel_step = processor.num_mel_frames_per_step

    chunks = [processor(audio[:first].numpy(), sampling_rate=SR,
                        is_streaming=True, is_first_audio_chunk=True)]
    mel_idx = mel_step
    start = processor.audio_chunk_start(mel_idx)
    while (end := start + per) <= audio.shape[0]:
        chunks.append(processor(audio[start:end].numpy(), sampling_rate=SR,
                                is_streaming=True, is_first_audio_chunk=False))
        mel_idx += mel_step
        start = processor.audio_chunk_start(mel_idx)
    chunks.append(processor(audio[start:].numpy(), sampling_rate=SR,
                            is_streaming=True, is_first_audio_chunk=False,
                            is_last_audio_chunk=True))

    cache = None
    parts = []
    for inp in chunks:
        inp = inp.to(device, dtype=model.dtype)
        out = model(**inp, speaker_cache=cache)
        parts.append(out.logits)
        cache = out.speaker_cache
    logits = torch.cat(parts, dim=1)
    return processor.extract_speaker_dict(logits)[0], logits, len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="offline",
                    choices=["offline", "low_latency", "very_low_latency", "ultra_low_latency"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.out and Path(args.out).exists() and not args.overwrite:
        print(f"skip existing {args.out}")
        return

    t0 = time.time()
    processor, model = load_model(args.device)
    load_s = time.time() - t0
    wav = load_audio_16k(args.audio)
    dur = len(wav) / SR
    t1 = time.time()
    if args.mode == "offline":
        segments, logits, n = diarize_chunked(processor, model, wav, device=args.device)
    else:
        segments, logits, n = diarize_streaming(processor, model, wav, args.mode, args.device)
    dt = time.time() - t1
    print(f"{Path(args.audio).name}: {dur/60:.2f} min, {n} chunks, {dt:.1f}s, "
          f"RTFx={dur/dt:.0f}, segments={len(segments)} (load {load_s:.1f}s)")

    if args.out:
        out = {
            "audio": str(args.audio),
            "duration": dur,
            "model": MODEL_ID,
            "mode": args.mode,
            "inference_seconds": dt,
            "rtfx": dur / dt,
            "segments": segments,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(args.out).with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
