"""Offline tests for scripts/pending.py (what the scoring loop waits on)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
import pending as P  # noqa: E402


class TestPending(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.saved = {k: getattr(C, k) for k in ("MANIFESTS", "SNIPPETS", "SCORES")}
        C.MANIFESTS = d / "manifests"
        C.SNIPPETS = d / "snippets"
        C.SCORES = d / "scores"
        for x in (C.MANIFESTS, C.SNIPPETS, C.SCORES):
            x.mkdir(parents=True)

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(C, k, v)
        self.tmp.cleanup()

    def manifest(self, rows):
        (C.MANIFESTS / "files.csv").write_text(
            "file_id,status\n" + "".join(f"{i},{s}\n" for i, s in rows),
            encoding="utf-8")

    def index(self, fid, n):
        (C.SNIPPETS / f"{fid}.index.jsonl").write_text(
            "".join(f'{{"segment_id":"{fid}_{k:05d}"}}\n' for k in range(n)),
            encoding="utf-8")

    def scored(self, fid, stage):
        (C.SCORES / f"{fid}.{stage}.jsonl").write_text(
            '{"segment_id":"x"}\n', encoding="utf-8")

    def test_counts_each_stage(self):
        self.manifest([("a", "ok"), ("b", "ok")])
        self.index("a", 2)
        self.index("b", 2)
        self.scored("a", "voicenet")
        self.scored("a", "empathic")
        self.assertEqual(P.pending(), [1, 1, 2])

    def test_empty_index_is_not_pending(self):
        self.manifest([("a", "ok"), ("b", "ok")])
        self.index("a", 2)
        self.index("b", 0)          # aligns to zero turns; can never be scored
        self.assertEqual(P.pending(), [1, 1, 1])

    def test_fixture_and_non_usable_rows_are_ignored(self):
        self.manifest([("a", "ok"), ("example", "example"), ("c", "failed")])
        self.index("a", 1)
        self.index("example", 3)
        self.index("c", 1)
        self.assertEqual(P.pending(), [1, 1, 1])

    def test_finished(self):
        self.manifest([("a", "ok")])
        self.index("a", 1)
        for stage in ("voicenet", "empathic", "voiceclap"):
            self.scored("a", stage)
        self.assertEqual(P.pending(), [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
