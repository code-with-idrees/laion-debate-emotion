"""Central config for the LAION Debate Emotion pipeline.

Every path is derived from ROOT, which defaults to the repository root and can be
overridden with the LAION_DEBATE_ROOT environment variable.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# Point LAION_DEBATE_ROOT elsewhere to keep data/models off the checkout.
ROOT = Path(os.environ.get("LAION_DEBATE_ROOT")
            or Path(__file__).resolve().parent.parent)
CACHE = ROOT / "cache" / "huggingface"
DATA = ROOT / "data"
AUDIO = DATA / "audio"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
DIAR = OUT / "diarization"
TRANSCRIPTS = OUT / "transcripts"
TURNS = OUT / "turns"
SNIPPETS = OUT / "snippets"
SCORES = OUT / "scores"
EMBEDDINGS = OUT / "embeddings"
DATASET = ROOT / "dataset"
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"
DOCS = ROOT / "docs"

for _p in (CACHE, DATA, AUDIO, MODELS, OUT, DIAR, TRANSCRIPTS, TURNS,
           SNIPPETS, SCORES, EMBEDDINGS, DATASET, MANIFESTS, LOGS, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

# --- Hugging Face ---------------------------------------------------------
DATASET_REPO = "teamsleeping/sleeping-debate"
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# HF_HUB_ENABLE_HF_TRANSFER is opt-in because xet handles our downloads.
if not os.environ.get("HF_TOKEN"):
    _tok = Path.home() / ".cache" / "huggingface" / "token"
    if _tok.exists():
        os.environ["HF_TOKEN"] = _tok.read_text(encoding="utf-8").strip()

# --- CUDA 12 DLLs for CTranslate2 (faster-whisper) ------------------------
# Installed by `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
# nvidia-cuda-runtime-cu12`. Located dynamically so the repo is portable.
_NV = None
_spec = importlib.util.find_spec("nvidia")
if _spec is not None:
    _locs = list(_spec.submodule_search_locations or [])
    _NV = Path(_locs[0]) if _locs else (Path(_spec.origin).parent if _spec.origin else None)
if _NV is not None:
    for _sub in ("cublas\\bin", "cudnn\\bin", "cuda_runtime\\bin"):
        _d = _NV / _sub
        if _d.exists():
            os.environ["PATH"] = str(_d) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(_d))
            except (AttributeError, OSError):
                pass

# --- Models ---------------------------------------------------------------
NEMOTRON_ID = "nvidia/Nemotron-3-Diarization"
WHISPER_CT2 = MODELS / "faster-whisper-large-v3"
WHISPER_CT2_TURBO = MODELS / "faster-whisper-large-v3-turbo"
VOICENET_DIR = MODELS / "voicenet"
# Empathic-Insight-Voice-Plus is a superset of Voice-Small (59 experts incl. the
# 4 audio-quality heads), so a single download covers the whole scorer.
EMPATHIC_PLUS_DIR = MODELS / "empathic_plus"
EMPATHIC_DIR = EMPATHIC_PLUS_DIR
BUDE_DIR = MODELS / "BUD-E-Whisper"
VOICECLAP_DIR = MODELS / "voiceclap-large-v2"

VOICECLAP_ID = "laion/voiceclap-large-v2"
EMPATHIC_ID = "laion/Empathic-Insight-Voice-Small"
EMPATHIC_PLUS_ID = "laion/Empathic-Insight-Voice-Plus"
BUDE_ID = "laion/BUD-E-Whisper"
VOICENET_ID = "laion/voicenet-dimension-predictors-commercial"

SR = 16000
FRAME_MS = 10
TURN_MAX_S = 30.0
TURN_MIN_S = 0.35
TURN_GAP_S = 0.8

# VoiceCLAP text probes (emotion + attribute + voice-quality vocabulary).
EMOTION_PROMPTS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
    "affection", "arousal", "awe", "contemplation", "contempt", "doubt",
    "fatigue", "hope", "impatience", "interest", "tension",
]
ATTRIBUTE_PROMPTS = [
    "confident and dominant voice", "hesitant and submissive voice",
    "warm and empathetic voice", "cold and detached voice",
    "serious and formal voice", "humorous and playful voice",
    "soft and soothing voice", "harsh and abrasive voice",
    "expressive and dynamic voice", "monotone and flat voice",
    "high-pitched voice", "low-pitched voice", "fast speaker", "slow speaker",
]
QUALITY_PROMPTS = [
    "clean studio recording", "noisy background", "reverberant room",
    "compressed telephone audio", "clear speech", "distant microphone",
]
VOICECLAP_PROMPTS = EMOTION_PROMPTS + ATTRIBUTE_PROMPTS + QUALITY_PROMPTS


def env_report() -> dict:
    return {
        "root": str(ROOT),
        "dataset": DATASET_REPO,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_token": bool(os.environ.get("HF_TOKEN")),
        "cuda_dlls": str(_NV),
    }
