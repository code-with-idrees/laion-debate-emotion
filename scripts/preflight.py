"""preflight.py - check the machine before starting a multi-day run.

Reports the Python/GPU/CUDA stack, free disk against the projected footprint,
how much of each model group is still missing, and whether the gated dataset is
reachable with the current token. Exits non-zero when something would stop the
pipeline, so it can guard a long run:

    python scripts/preflight.py || exit 1

Nothing is downloaded or modified here (a single 1-byte Range request is made
to test dataset access).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C

OK, WARN, BAD = "ok  ", "warn", "FAIL"


def head(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def env_checks() -> list[tuple[str, str, str]]:
    rows = []
    rows.append(("python", OK, f"{sys.version.split()[0]} ({sys.executable})"))
    rows.append(("root", OK if C.ROOT.exists() else BAD, str(C.ROOT)))

    try:
        import torch
        rows.append(("torch", OK, torch.__version__))
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            rows.append(("cuda", OK, f"{name}, {vram:.1f} GB, CUDA {torch.version.cuda}"))
        else:
            rows.append(("cuda", BAD, "torch.cuda.is_available() is False"))
    except Exception as e:  # pragma: no cover - environment dependent
        rows.append(("torch", BAD, f"{type(e).__name__}: {e}"))

    for mod in ("transformers", "sentence_transformers", "faster_whisper",
                "pyarrow", "soundfile", "numpy"):
        try:
            m = __import__(mod)
            rows.append((mod, OK, getattr(m, "__version__", "?")))
        except Exception as e:  # pragma: no cover
            rows.append((mod, BAD, f"not importable ({type(e).__name__})"))

    rows.append(("hf token", OK if os.environ.get("HF_TOKEN") else WARN,
                 "found" if os.environ.get("HF_TOKEN") else
                 "missing (gated dataset and some models need it)"))
    return rows


def disk_check() -> list[tuple[str, str, str]]:
    free = shutil.disk_usage(C.ROOT).free
    # 436 GB audio + ~150 GB FLAC snippets + 40 GB models + 10 GB embeddings
    need = 636 * 1024**3
    return [("disk free", OK if free > need else WARN,
             f"{free / 1024**3:.0f} GB free, ~{need / 1024**3:.0f} GB projected "
             f"(downloads can be pruned after scoring)")]


def partial_bytes(local: Path, name: str, size: int) -> int:
    """Bytes already checkpointed for an in-flight download (see hf_fetch)."""
    import json

    state = local / (name + ".blocks.json")
    if not state.exists():
        return 0
    try:
        st = json.loads(state.read_text())
        return min(size, len(st.get("done") or []) * int(st.get("block") or 0))
    except Exception:
        return 0


def model_check() -> list[tuple[str, str, str]]:
    """Ask the Hub which files of each group are still missing."""
    from huggingface_hub import HfApi

    spec = importlib.util.spec_from_file_location(
        "_dl", Path(__file__).parent / "00_download_models.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    api = HfApi()
    rows = []
    for name, (repo, local, patterns) in dl.GROUPS.items():
        try:
            todo, total, have = dl.list_pending(api, repo, local, patterns)
        except Exception as e:  # network / auth
            rows.append((name, BAD, f"cannot query Hub: {type(e).__name__}: {e}"))
            continue
        pct = 100.0 * have / total if total else 100.0
        if todo:
            partial = sum(partial_bytes(local, t["name"], t["size"]) for t in todo)
            pct = 100.0 * (have + partial) / total if total else 100.0
            extra = f" ({(have + partial) / 1e9:.1f} GB incl. partial)" if partial else ""
            rows.append((name, WARN,
                         f"{have / 1e9:.1f}/{total / 1e9:.1f} GB ({pct:.0f}%){extra}, "
                         f"{len(todo)} file(s) missing -> models/{local.name}"))
        else:
            rows.append((name, OK, f"{total / 1e9:.1f} GB complete"))
    return rows


def dataset_check() -> list[tuple[str, str, str]]:
    import requests
    from huggingface_hub import HfApi

    api = HfApi()
    rows = []
    try:
        info = api.dataset_info(C.DATASET_REPO)
        gated = getattr(info, "gated", None) or "false"
        rows.append(("dataset", OK, f"{C.DATASET_REPO} (gated={gated})"))
        files = [s.rfilename for s in info.siblings if s.rfilename.endswith(".wav")]
        rows.append(("dataset files", OK, f"{len(files)} wav files listed"))
        if not files:
            return rows
        # Range request for one byte tells us whether *we* may read the data.
        url = (f"https://huggingface.co/datasets/{C.DATASET_REPO}"
               f"/resolve/main/{files[0]}")
        r = requests.get(url, headers={
            "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
            "Range": "bytes=0-0"}, timeout=(15, 30), allow_redirects=True)
        if r.status_code in (200, 206):
            rows.append(("dataset access", OK, "1-byte read succeeded"))
        else:
            rows.append(("dataset access", BAD,
                         f"HTTP {r.status_code} - request access on the Hub"))
    except Exception as e:  # pragma: no cover - network dependent
        rows.append(("dataset", BAD, f"{type(e).__name__}: {e}"))
    return rows


def print_rows(rows):
    width = max(len(r[0]) for r in rows) if rows else 0
    for name, status, detail in rows:
        print(f"  [{status}] {name:<{width}}  {detail}")


def main() -> int:
    print(f"preflight for {C.ROOT}")
    head("environment")
    env = env_checks()
    print_rows(env)

    head("disk")
    disk = disk_check()
    print_rows(disk)

    head("models")
    models = model_check()
    print_rows(models)

    head("dataset")
    data = dataset_check()
    print_rows(data)

    blockers = [r for r in env + disk + models + data if r[1] == BAD]
    print()
    if blockers:
        print(f"{len(blockers)} blocker(s):")
        for name, _, detail in blockers:
            print(f"  - {name}: {detail}")
        return 1

    missing_models = [r for r in models if r[1] == WARN]
    if missing_models:
        print("ready to continue downloads: python scripts/00_download_models.py")
    else:
        print("ready: python scripts/01_download_dataset.py --workers 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
