"""Offline test for the manifest counter used by the download supervisor."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import manifest_ok  # noqa: E402


class TestCounts(unittest.TestCase):
    def test_missing_file(self):
        self.assertEqual(manifest_ok.counts(Path("does-not-exist.csv")), {})

    def test_groups_by_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "files.csv"
            p.write_text(
                "file_id,status\n"
                "a,ok\n"
                "b,ok\n"
                "c,error\n"
                "d,skip\n",
                encoding="utf-8")
            c = manifest_ok.counts(p)
        self.assertEqual(c["ok"], 2)
        self.assertEqual(c["error"], 1)
        self.assertEqual(c["skip"], 1)
        self.assertEqual(c["total"], 4)


if __name__ == "__main__":
    unittest.main()
