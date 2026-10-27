"""Offline tests for the CPU-stage parallel helper."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import parallel  # noqa: E402


def worker(row, overwrite):
    if row["file_id"] == "boom":
        raise RuntimeError("deliberate failure")
    return {"file_id": row["file_id"], "overwrite": overwrite}


ROWS = [{"file_id": "a"}, {"file_id": "b"}, {"file_id": "boom"}, {"file_id": "d"}]


class TestRunRows(unittest.TestCase):
    def collect(self, jobs):
        return list(parallel.run_rows(ROWS, worker, jobs, overwrite=True))

    def check(self, out):
        self.assertEqual(len(out), 4)
        for i, total, row, status, payload in out:
            self.assertEqual(total, 4)
            if row["file_id"] == "boom":
                self.assertEqual(status, "error")
                self.assertIn("deliberate failure", payload)
            else:
                self.assertEqual(status, "ok")
                self.assertTrue(payload["overwrite"])
        self.assertEqual({r["file_id"] for _, _, r, _, _ in out}, {"a", "b", "boom", "d"})

    def test_serial(self):
        self.check(self.collect(1))

    def test_parallel(self):
        self.check(self.collect(2))

    def test_zero_jobs_means_auto(self):
        self.assertGreaterEqual(parallel.default_jobs(), 1)
        out = list(parallel.run_rows(ROWS[:2], worker, 0))
        # Parallel workers finish in completion order, not submission order.
        self.assertEqual({r["file_id"] for _, _, r, _, _ in out}, {"a", "b"})

    def test_empty(self):
        self.assertEqual(list(parallel.run_rows([], worker, 4)), [])

    def test_single_row_short_circuits(self):
        out = list(parallel.run_rows(ROWS[:1], worker, 4))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][3], "ok")


if __name__ == "__main__":
    unittest.main()
