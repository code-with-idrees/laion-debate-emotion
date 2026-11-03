"""Offline unit tests for the shared scoring helpers (score_common)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import score_common as S  # noqa: E402


def rows(*durations):
    return [{"segment_id": f"s{i:03d}", "duration": d, "path": f"s{i:03d}.flac"}
            for i, d in enumerate(durations)]


class TestMakeBatches(unittest.TestCase):
    def test_count_limit(self):
        batches = list(S.make_batches(rows(*[1.0] * 7), max_batch=3,
                                      max_seconds=100))
        self.assertEqual([len(b) for b in batches], [3, 3, 1])

    def test_duration_limit(self):
        batches = list(S.make_batches(rows(4.0, 4.0, 4.0), max_batch=10,
                                      max_seconds=10.0))
        self.assertEqual([len(b) for b in batches], [2, 1])

    def test_single_long_item_still_emitted(self):
        batches = list(S.make_batches(rows(50.0), max_batch=2, max_seconds=10))
        self.assertEqual([len(b) for b in batches], [1])

    def test_missing_duration_is_zero(self):
        items = [{"segment_id": "a", "path": "a"}, {"segment_id": "b", "path": "b"}]
        batches = list(S.make_batches(items, max_batch=5, max_seconds=10))
        self.assertEqual([len(b) for b in batches], [2])

    def test_empty(self):
        self.assertEqual(list(S.make_batches([], 4, 10)), [])


class TestJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_atomic_write_and_read(self):
        p = self.dir / "out.jsonl"
        S.atomic_write_jsonl(p, [{"segment_id": "a", "x": 1},
                                 {"segment_id": "b", "x": 2}])
        self.assertEqual(S.existing_ids(p), ["a", "b"])
        self.assertFalse(list(self.dir.glob("*.tmp")))

    def test_missing_file(self):
        self.assertIsNone(S.existing_ids(self.dir / "nope.jsonl"))

    def test_corrupt_file_returns_none(self):
        p = self.dir / "bad.jsonl"
        p.write_text("{not json}\n", encoding="utf-8")
        self.assertIsNone(S.existing_ids(p))

    def test_partial_file_still_parses(self):
        p = self.dir / "partial.jsonl"
        p.write_text(json.dumps({"segment_id": "a"}) + "\n", encoding="utf-8")
        self.assertEqual(S.existing_ids(p), ["a"])


class TestBatchErrors(unittest.TestCase):
    """One unreadable snippet must not kill a stage that has 200k to get through."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "bad_snippets.log"
        self._saved = S.BAD_SNIPPETS_LOG
        S.BAD_SNIPPETS_LOG = self.log

    def tearDown(self):
        S.BAD_SNIPPETS_LOG = self._saved
        self.tmp.cleanup()

    def test_bad_item_is_isolated_and_logged(self):
        items = rows(1.0, 1.0, 1.0, 1.0)

        def batch(sub):
            if any(x["path"] == "s002.flac" for x in sub):
                raise ValueError("array is too big")
            return [{"v": 1} for _ in sub]

        out = S.score_with_retry(batch, items)
        self.assertEqual(len(out), 4)
        self.assertEqual(sum(1 for r in out if "error" in r), 1)
        self.assertIn("s002.flac", self.log.read_text(encoding="utf-8"))

    def test_good_batches_untouched(self):
        def batch(sub):
            return [{"v": len(sub)} for _ in sub]

        self.assertEqual(len(S.score_with_retry(batch, rows(1.0, 1.0))), 2)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
