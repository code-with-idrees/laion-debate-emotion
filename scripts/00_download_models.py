"""00 - Download all model weights into <root>/models (see config.py).

Robust ranged downloader (see hf_fetch.py): per-block Range requests with
socket timeouts, block-level resume checkpoints, and partial reuse of an
existing huggingface_hub .incomplete blob (verified by sha256 when used).

Examples
--------
  python 00_download_models.py                  # every group, 4 files in flight
  python 00_download_models.py --only voicenet
  python 00_download_models.py --workers 4 --chunk-threads 4
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from hf_fetch import Downloader
from huggingface_hub import HfApi

GROUPS = {
    "voicenet": (C.VOICENET_ID, C.VOICENET_DIR, None),
    "bude": (C.BUDE_ID, C.BUDE_DIR, None),
    "empathic": (C.EMPATHIC_PLUS_ID, C.EMPATHIC_PLUS_DIR,
                 ["model_*.pth", "*.json", "README.md"]),
    "voiceclap": (C.VOICECLAP_ID, C.VOICECLAP_DIR, None),
}

_lock = threading.RLock()
STATE_P = C.LOGS / "dl_state.json"


def log(msg: str):
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
        _write_state()


_state = {"group": None, "files_done": 0, "files_total": 0,
          "bytes_done": 0, "bytes_total": 0, "current": [], "ts": None}


def _write_state():
    _state["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATE_P.with_suffix(".tmp")
    tmp.write_text(json.dumps(_state, indent=2))
    tmp.replace(STATE_P)


def matches(name, patterns) -> bool:
    return True if not patterns else any(fnmatch.fnmatch(name, p) for p in patterns)


def list_pending(api, repo, local: Path, patterns):
    info = api.model_info(repo, files_metadata=True, token=os.environ.get("HF_TOKEN"))
    todo, total, have = [], 0, 0
    for s in info.siblings:
        if not matches(s.rfilename, patterns):
            continue
        size = int(s.size or 0)
        total += size
        dest = local / s.rfilename
        if dest.exists() and (size == 0 or dest.stat().st_size == size):
            have += size
            continue
        sha = None
        if s.lfs is not None:
            sha = getattr(s.lfs, "sha256", None)
        todo.append({"name": s.rfilename, "size": size, "sha256": sha})
    return todo, total, have


def seed_from_incomplete(dest: Path, size: int):
    """Reuse a huggingface_hub partial download as the .part prefix.

    Returns (part_path, prefix_len) or None.
    """
    cache = dest.parent / ".cache" / "huggingface" / "download"
    if not cache.exists():
        return None
    best = None
    for f in cache.rglob("*.incomplete"):
        n = f.stat().st_size
        if 0 < n < size and (best is None or n > best.stat().st_size):
            best = f
    if best is None:
        return None
    prefix = best.stat().st_size
    part = dest.with_name(dest.name + ".part")
    if part.exists() and part.stat().st_size >= prefix:
        return part, prefix
    shutil.copyfile(best, part)
    with open(part, "r+b") as fh:
        fh.truncate(size)
    return part, prefix


def sha256_of(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        left = limit_bytes
        while True:
            n = 1 << 22 if left is None else min(1 << 22, left)
            if n == 0:
                break
            b = fh.read(n)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def run_group(api, dl: Downloader, key: str, workers: int, max_rounds: int):
    repo, local, patterns = GROUPS[key]
    local.mkdir(parents=True, exist_ok=True)
    log(f"=== {key}: {repo} -> {local}")

    for rnd in range(1, max_rounds + 1):
        todo, total, have = list_pending(api, repo, local, patterns)
        with _lock:
            _state.update({"group": key, "files_total": len(todo),
                           "files_done": 0, "bytes_total": total - have,
                           "bytes_done": 0})
        if not todo:
            log(f"=== {key}: complete ({total/1e9:.2f} GB)")
            return True
        log(f"=== {key}: round {rnd}, {len(todo)} files, "
            f"{(total-have)/1e9:.2f} GB missing of {total/1e9:.2f} GB")

        t0 = time.time()
        ok_bytes = {"n": 0}
        counter = {"n": 0}

        def one(item):
            name, size, sha = item["name"], item["size"], item["sha256"]
            dest = local / name
            seeded = None
            prefix = 0
            if size and size > 100 * 1024 * 1024:
                got = seed_from_incomplete(dest, size)
                if got:
                    seeded, prefix = got
            good, msg = dl.download(repo, name, dest, size, assume_prefix=prefix)
            if good and sha and msg != "present":
                if sha256_of(dest) != sha:
                    log(f"  !! sha256 mismatch {name}; restarting")
                    dest.unlink(missing_ok=True)
                    (dest.with_name(dest.name + ".part")).unlink(missing_ok=True)
                    (dest.with_name(dest.name + ".blocks.json")).unlink(missing_ok=True)
                    good, msg = dl.download(repo, name, dest, size)
                    if good and sha256_of(dest) != sha:
                        good, msg = False, "sha256 mismatch"
            with _lock:
                counter["n"] += 1
                _state["files_done"] = counter["n"]
                if good:
                    ok_bytes["n"] += size
                    _state["bytes_done"] = ok_bytes["n"]
                rate = ok_bytes["n"] / max(time.time() - t0, 1e-6) / 1e6
            log(f"  [{counter['n']}/{len(todo)}] {name} "
                f"{'OK' if good else 'FAIL'} {msg} | agg {rate:.2f} MB/s")
            return good

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, it) for it in todo]
            for f in as_completed(futs):
                f.result()

        left, _, _ = list_pending(api, repo, local, patterns)
        if not left:
            log(f"=== {key}: complete")
            return True
        log(f"=== {key}: {len(left)} files still missing, next round")
    log(f"=== {key}: INCOMPLETE")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list(GROUPS))
    ap.add_argument("--workers", type=int, default=4, help="files in parallel")
    ap.add_argument("--chunk-threads", type=int, default=4, help="connections per file")
    ap.add_argument("--block-mb", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    dl = Downloader(os.environ.get("HF_TOKEN"), block_mb=args.block_mb,
                    chunk_threads=args.chunk_threads, timeout=args.timeout)

    # progress monitor
    stop = threading.Event()

    def monitor():
        last_n, last_t = dl.done_bytes, time.time()
        while not stop.wait(20):
            n, t = dl.done_bytes, time.time()
            dt = max(t - last_t, 1e-6)
            log(f"    ... {n/1e9:.2f} GB this session "
                f"({(n-last_n)/dt/1e6:.2f} MB/s)")
            last_n, last_t = n, t

    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    try:
        results = {}
        for key in (args.only or list(GROUPS)):
            results[key] = run_group(api, dl, key, args.workers, args.rounds)
    finally:
        stop.set()
    log(f"summary: {results}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
