"""Offline tests for 05_cut_snippets selection (no audio decoding).

The real bug these cover: the driver hands `--limit 20` to every stage, and 05
sliced the manifest *before* dropping already-cut and skip-marked recordings, so
each round re-examined the same finished files and the pending count never fell.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_stage():
    spec = importlib.util.spec_from_file_location(
        "_script_05", ROOT / "scripts" / "05_cut_snippets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_stage()


class TestCutFile(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        self._saved = (S.C.TURNS, S.C.SNIPPETS)
        S.C.TURNS = root / "turns"
        S.C.SNIPPETS = root / "snippets"
        for d in (S.C.TURNS, S.C.SNIPPETS):
            d.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        S.C.TURNS, S.C.SNIPPETS = self._saved
        self.tmp.cleanup()

    def test_skip_marker_is_not_pending(self):
        (S.C.TURNS / "x.skip.json").write_text(json.dumps({"file_id": "x"}))
        self.assertIsNone(S.cut_file({"file_id": "x"}))

    def test_missing_turns_is_reported(self):
        self.assertEqual(S.cut_file({"file_id": "y"}),
                         {"file_id": "y", "status": "missing_turns"})

    def test_existing_index_is_skipped(self):
        (S.C.SNIPPETS / "z.index.jsonl").write_text("")
        self.assertIsNone(S.cut_file({"file_id": "z"}))


if __name__ == "__main__":
    unittest.main()
