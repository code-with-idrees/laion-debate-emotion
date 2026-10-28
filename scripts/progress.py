"""progress.py - live dashboard for the 436 GB corpus pull.

    python scripts\\progress.py
    python scripts\\progress.py --once      # single snapshot, for logs/cron

Refreshes once a second until Ctrl-C. Downloaded bytes are counted from the
16 MB block checkpoints of in-flight files plus the finished WAVs, because the
`.part` files are preallocated and their size says nothing about progress.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path

import config as C

WINDOW_S = 90


def enable_ansi() -> None:
    """Turn on VT sequences in the legacy Windows console (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
    except Exception:
        pass


def total_gb() -> float:
    """Corpus size, read from the downloader's banner when it is available."""
    log = C.LOGS / "dl_dataset.log"
    try:
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "size=" in line and "GB" in line:
                return float(line.split("size=")[1].split("GB")[0])
    except OSError:
        pass
    return 436.42


def done_bytes(audio: Path | None = None) -> int:
    """Finished WAVs plus the bytes already fetched for in-flight files.

    A file's blocks.json counts only until its WAV exists (the checkpoint is
    deleted on completion, but the two can briefly overlap), and the last block
    of a file is usually partial: counting ``len(done) * block`` overstates it,
    which made the total *drop* when a file completed and the rolling rate read
    0.00 MB/s while the download was actually moving at 7 MB/s.
    """
    audio = audio or C.AUDIO
    total = 0
    for p in audio.glob("*.wav"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    for p in audio.glob("*.wav.blocks.json"):
        base = p.name[: -len(".blocks.json")]
        if (audio / base).exists():
            continue  # finished; the WAV already counts it
        part = audio / (base + ".part")
        try:
            size = part.stat().st_size
            st = json.loads(p.read_text())
        except Exception:
            continue
        block = int(st.get("block", 0)) or 1
        for i in st.get("done", []):
            total += min(block, max(0, size - int(i) * block))
    return total


def manifest(path: Path | None = None) -> dict[str, int]:
    path = path or (C.MANIFESTS / "files.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, int] = {"total": len(rows)}
    for r in rows:
        k = r.get("status") or "unknown"
        out[k] = out.get(k, 0) + 1
    return out


def bar(frac: float, width: int = 28) -> str:
    n = max(0, min(width, int(width * frac)))
    return "#" * n + "." * (width - n)


def human(seconds: float) -> str:
    if seconds != seconds or seconds <= 0:  # NaN or negative
        return "?"
    d, rem = divmod(int(seconds), 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    return f"{d}d {h:02d}h {m:02d}m" if d else f"{h:02d}h {m:02d}m"


def frame(got: int, total_bytes: float, rate: float, have: int,
          sizes: dict[str, int], window: int = WINDOW_S) -> list[str]:
    remaining = max(0.0, total_bytes - got)
    eta = remaining / rate if rate > 0 else float("nan")
    diar = sum(1 for _ in C.DIAR.glob("*.json"))
    snippets = sum(1 for _ in C.SNIPPETS.glob("*/*.flac"))
    return [
        f"corpus      {C.AUDIO}",
        f"download    [{bar(have / C.DATASET_N_FILES)}] {have:5d}/{C.DATASET_N_FILES} files   "
        f"{got / 1e9:7.2f}/{total_bytes / 1e9:.1f} GB "
        f"({100 * got / total_bytes:5.2f}%)",
        f"speed       {rate / 1e6:5.2f} MB/s (last {window}s)   "
        f"ETA {human(eta)}   errors {sizes.get('error', 0)}",
        f"pipeline    diarized {diar:5d}   snippets {snippets:7,d}",
        f"updated     {time.strftime('%H:%M:%S')}   Ctrl-C stops watching, "
        f"not the download",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="print one snapshot")
    ap.add_argument("--frames", type=int, default=0, help="stop after N refreshes")
    args = ap.parse_args()

    total_bytes = total_gb() * 1e9
    window: deque[tuple[float, int]] = deque()
    frames = 0
    if args.once:
        # A single read cannot show a speed: sample 5 s so a snapshot in a log
        # says whether the pull is actually moving.
        before, t0 = done_bytes(), time.time()
        time.sleep(5.0)
        got = done_bytes()
        rate = (got - before) / max(time.time() - t0, 1e-6)
        sizes = manifest()
        print("\n".join(frame(got, total_bytes, rate, sizes.get("ok", 0),
                             sizes, window=5)))
        return 0
    enable_ansi()
    print("\033[2J", end="")

    while True:
        now = time.time()
        got = done_bytes()
        window.append((now, got))
        while len(window) > 1 and now - window[0][0] > WINDOW_S:
            window.popleft()
        span = now - window[0][0]
        rate = (got - window[0][1]) / span if span > 5 else 0.0
        sizes = manifest()
        have = sizes.get("ok", 0)
        lines = frame(got, total_bytes, rate, have, sizes)

        print("\033[H" + "\n".join(line + "\033[K" for line in lines), end="",
              flush=True)
        frames += 1
        if args.frames and frames >= args.frames:
            print()
            return 0
        time.sleep(1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped watching; the download keeps running.")
