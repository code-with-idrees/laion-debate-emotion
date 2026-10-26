"""Parallel execution helper for CPU-bound per-file stages.

Turn building, snippet cutting and assembly are independent per recording, so
they are spread over all available cores. Each worker opens its own files and
writes only its own outputs; the parent process is the only one printing, so
progress lines stay readable and Windows spawn semantics are safe.
"""
from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed


def default_jobs() -> int:
    """All logical cores; stages that also feed a GPU can pass --jobs 2."""
    return max(1, os.cpu_count() or 1)


def _call(worker, row, overwrite):
    try:
        return "ok", worker(row, overwrite)
    except Exception:
        return "error", traceback.format_exc()


def run_rows(rows, worker, jobs: int = 0, overwrite: bool = False):
    """Yield ``(i, total, row, status, payload)`` for every manifest row.

    ``worker(row, overwrite)`` must be a module-level function (it has to be
    picklable) and should return a stats dict, or ``None`` when it skipped the
    file. Exceptions are captured and yielded with ``status == "error"`` and the
    formatted traceback as payload, so one bad recording cannot kill the run.
    """
    total = len(rows)
    jobs = default_jobs() if not jobs else max(1, int(jobs))

    if jobs == 1 or total <= 1:
        for i, row in enumerate(rows, 1):
            status, payload = _call(worker, row, overwrite)
            yield i, total, row, status, payload
        return

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(_call, worker, row, overwrite): row for row in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            status, payload = fut.result()
            yield i, total, futures[fut], status, payload
