"""hf_fetch.py - resumable, timeout-guarded ranged downloader for HF repos.

Why not hf_hub_download: on this connection the transfer occasionally stalls
mid-file with no progress and no timeout, leaving a huge .incomplete blob.
Here every file is split into fixed-size blocks, each block is a separate
Range request with a socket timeout, completed blocks are checkpointed to a
sidecar JSON, and a stalled block is simply retried.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://huggingface.co"
BLOCK_MB = 16
USER_AGENT = "laion-debate-emotion-pipeline/1.0"


class Downloader:
    def __init__(self, token: str | None, block_mb: int = BLOCK_MB,
                 chunk_threads: int = 4, timeout: int = 30, retries: int = 6):
        self.token = token or ""
        self.block = int(block_mb * 1024 * 1024)
        self.chunk_threads = chunk_threads
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()
        self.done_bytes = 0
        self.done_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers.update({
                "Authorization": f"Bearer {self.token}",
                "User-Agent": USER_AGENT,
            })
            self._local.session = s
        return s

    def _url(self, repo: str, path: str, repo_type: str = "model") -> str:
        prefix = "" if repo_type == "model" else f"{repo_type}s/"
        return f"{BASE}/{prefix}{repo}/resolve/main/{path}?download=true"

    # ------------------------------------------------------------------ #
    def _fetch_block(self, repo: str, path: str, part: Path, start: int,
                     end: int, repo_type: str = "model"):
        """Download [start, end] inclusive into part at the right offset."""
        want = end - start + 1
        for attempt in range(self.retries):
            try:
                r = self._session().get(
                    self._url(repo, path, repo_type),
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True, timeout=(15, self.timeout),
                )
                if r.status_code == 200:
                    # Server ignored Range: cannot resume this block.
                    raise RuntimeError("server ignored Range (200)")
                if r.status_code not in (206, 416):
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code == 416:
                    return True  # already satisfied
                buf = bytearray()
                for piece in r.iter_content(chunk_size=512 * 1024):
                    if piece:
                        buf.extend(piece)
                        if len(buf) > want:
                            break
                if len(buf) != want:
                    raise RuntimeError(f"short block {len(buf)}/{want}")
                with open(part, "r+b") as fh:
                    fh.seek(start)
                    fh.write(buf)
                with self.done_lock:
                    self.done_bytes += want
                return True
            except Exception:
                if attempt == self.retries - 1:
                    return False
                time.sleep(min(30, 2 * (attempt + 1)))
        return False

    # ------------------------------------------------------------------ #
    def _save_state(self, state_p: Path, done: set[int]) -> None:
        """Atomically checkpoint finished blocks.

        A plain write_text() leaves truncated JSON when the process is killed,
        and truncated state made the next run re-download all 1,064 blocks of an
        18 GB file - expensive at 1 MB/s over a flaky link.
        """
        tmp = state_p.with_name(state_p.name + ".tmp")
        tmp.write_text(json.dumps({"block": self.block, "done": sorted(done)}))
        os.replace(tmp, state_p)

    def _load_state(self, state_p: Path, n_blocks: int) -> set[int]:
        """Read the checkpoint, salvaging indices from a truncated file."""
        try:
            st = json.loads(state_p.read_text())
        except Exception:
            # Killed mid-write: blocks recorded before the truncation were
            # written to the .part file first, so they are still good. The
            # final number may be a partial index ("..., 23" cut to ", 2"), so
            # drop it unless the JSON happened to end cleanly.
            text = state_p.read_text(errors="ignore")
            tail = text.split("[", 1)[-1]
            nums = [int(x) for x in re.findall(r"\d+", tail)]
            if not text.rstrip().endswith("]") and nums:
                nums = nums[:-1]
            return {i for i in nums if 0 <= i < n_blocks}
        if isinstance(st, dict) and st.get("block") == self.block:
            return {int(i) for i in st.get("done", [])}
        if isinstance(st, list) and self.block == 16 * 1024 * 1024:
            # legacy format, written by the first version (16 MB blocks)
            return {int(i) for i in st}
        return set()

    def download(self, repo: str, path: str, dest: Path, size: int,
                 assume_prefix: int = 0,
                 repo_type: str = "model") -> tuple[bool, str]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and (size == 0 or dest.stat().st_size == size):
            return True, "present"

        part = dest.with_name(dest.name + ".part")
        state_p = dest.with_name(dest.name + ".blocks.json")
        n_blocks = max(1, -(-size // self.block)) if size else 1
        done = set()
        if part.exists() and part.stat().st_size == size and state_p.exists():
            done = self._load_state(state_p, n_blocks)
        if not part.exists() or part.stat().st_size != size:
            # fresh (or size changed): preallocate
            with open(part, "wb") as fh:
                if size:
                    fh.truncate(size)
            done = set()
        if not done and assume_prefix > 0:
            # a verified prefix was seeded: every fully-covered block is done
            done = {i for i in range(n_blocks)
                    if (i + 1) * self.block <= assume_prefix}
            self._save_state(state_p, done)

        todo = [i for i in range(n_blocks) if i not in done]
        if not todo:
            os.replace(part, dest)
            state_p.unlink(missing_ok=True)
            return True, "resumed-complete"

        t0 = time.time()
        failed = []

        def job(i):
            start = i * self.block
            end = min(size - 1, start + self.block - 1) if size else 0
            return i, self._fetch_block(repo, path, part, start, end, repo_type)

        with ThreadPoolExecutor(max_workers=self.chunk_threads) as ex:
            futs = [ex.submit(job, i) for i in todo]
            for f in as_completed(futs):
                i, ok = f.result()
                if ok:
                    done.add(i)
                    self._save_state(state_p, done)
                else:
                    failed.append(i)

        if failed or len(done) < n_blocks:
            return False, f"failed blocks {len(failed or done)}"

        if size and part.stat().st_size != size:
            return False, f"size {part.stat().st_size} != {size}"
        os.replace(part, dest)
        state_p.unlink(missing_ok=True)
        return True, f"{size/1e6:.1f} MB in {time.time()-t0:.0f}s"


def probe_size(dl: Downloader, repo: str, path: str) -> int:
    r = dl._session().head(dl._url(repo, path), allow_redirects=True,
                           timeout=(15, 30))
    return int(r.headers.get("content-length") or 0)
