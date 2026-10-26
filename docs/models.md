# Models used

Four models run over the corpus. All weight files are downloaded by
`scripts/00_download_models.py` into `models/`; nothing is fetched at inference
time. Raw model cards are kept out of the repository (`docs/vendor/` is
gitignored); this page records what the pipeline relies on.

| stage | repo | size | licence |
|-------|------|------|---------|
| diarization | `nvidia/Nemotron-3-Diarization` | ~0.5 GB | OpenMDW-1.1 |
| ASR | `Systran/faster-whisper-large-v3` (CT2) | ~3 GB | MIT |
| empathic | `laion/Empathic-Insight-Voice-Plus` + `laion/BUD-E-Whisper` | ~16 GB + 1 GB | see card |
| voicenet | `laion/voicenet-dimension-predictors-commercial` | ~0.5 GB | CC-BY-4.0 |
| voiceclap | `laion/voiceclap-large-v2` | ~18 GB | CC-BY-4.0 |

## Nemotron 3 Diarization

Frame-classification model with a recurrent speaker cache. Encoder frames are
80 ms; the processor can stream in 1.04 s steps or score long windows at once.
Up to 8 speakers, English broadcast/debate style training data.

`nemotron_diarize.py` uses the windowed path: 27.2 s of newly scored audio per
forward pass, 3.2 s of right context so a speaker turn that straddles a window
boundary is still resolved, and `speaker_cache` passed between windows. The
cache is what makes windows independent instead of a fresh clustering problem
every 30 s. Output segments are `{Start, End, Speaker}` in seconds.

## faster-whisper large-v3

CTranslate2 float16 conversion of Whisper large-v3. The pipeline needs word
timestamps, so `word_timestamps=True`; `vad_filter=True` removes silence before
decoding and `condition_on_previous_text=False` stops one hallucinated segment
from poisoning the rest of a 20-minute debate. Batched inference
(`BatchedInferencePipeline`, batch 16) gives ~36x realtime on the 3090.

## Empathic-Insight-Voice-Plus

A BUD-E-Whisper encoder (88 M parameters, 768-d hidden) with **59 linear
experts** on top:

- 55 affect / attribute / pitch experts, identical architecture
  (`1500x768 -> 64 -> 64 -> 32 -> 16 -> 1`, ReLU), trained on
  `Empathic-Insight-Voice-Small`;
- 4 audio-quality heads (`background`, `content`, `overall`, `speech`) that use
  mean + min + max + std pooling over time (3072-d input).

Each expert is a separate `model_*_best.pth` checkpoint, so a naive
implementation loads 59 copies of a 74 M-parameter module. Instead the 55
projection matrices are fused into one fp16 bank and applied as batched
matmuls; the quality heads reuse the same encoder output with the 4-statistic
pool. All 59 experts run in one pass per batch of snippets.

Output: `{"empathic": {label: score}, "empathic_top5": [...]}`.

## VoiceNet dimension predictors

57 psychological / vocal dimensions (arousal, valence, age, gender, roughness,
speech rate, ...), each with a regression head and a 5-way bucket classifier,
plus a confidence output. Input is a `voiceclap-commercial` embedding
(768-d, shipped inside the same repo) — **not** a raw waveform — so the scorer
encodes snippets with the small commercial model first.

Bucket labels are full sentences ("not at all aroused ... extremely aroused"),
so they are written once to `dataset/voicenet_levels.json` and rows keep only
the bucket index.

Output: `{"voicenet": {"reg": {...}, "cls": {...}, "conf": {...}}}`.

## VoiceCLAP-large-v2

A rank-16 LoRA finetune (`alpha=32`, all-linear) of **LCO-Embedding-Omni-7B**
(Qwen2.5-Omni thinker) trained with prototypical contrastive learning, so audio
and text map into the same 3584-d space. 7 B parameters in bfloat16 fit in
24 GB only if the encoder is loaded once and batching stays small.

The pipeline encodes each snippet once, L2-normalises the embedding and stores
it in a per-file fp16 `.npy` sidecar (one matrix per source recording). The
table keeps the top-5 text-probe similarities for a fixed vocabulary of 59
prompts (39 emotions, 14 voice attributes, 6 quality descriptors) plus the row
offset into the `.npy`. That keeps the Parquet small while preserving the full
embedding for anyone who wants to run their own probes.
