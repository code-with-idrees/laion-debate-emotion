"""Offline tests for the assembly join (script 09).

Paths in config are module attributes that every stage reads at call time, so
the test can point TURNS/SCORES/EMBEDDINGS/DATASET at a temporary directory and
exercise the real join without touching the project outputs.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402


def load_assembly():
    spec = importlib.util.spec_from_file_location(
        "_script_09", ROOT / "scripts" / "09_assemble_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = load_assembly()

TURNS = [
    {"segment_id": "f_00000", "file_id": "f", "start": 0.0, "end": 1.0,
     "duration": 1.0, "speaker": "speaker_0", "text": "hello",
     "n_words": 1, "overlap": 0.0, "asr_confidence": 0.9},
    {"segment_id": "f_00001", "file_id": "f", "start": 1.0, "end": 2.5,
     "duration": 1.5, "speaker": "speaker_1", "text": "hi there",
     "n_words": 2, "overlap": 0.1, "asr_confidence": 0.8},
]


class TestCollectRows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.saved = {k: getattr(C, k) for k in
                      ("TURNS", "SCORES", "EMBEDDINGS", "DATASET")}
        C.TURNS = self.dir / "turns"
        C.SCORES = self.dir / "scores"
        C.EMBEDDINGS = self.dir / "emb"
        C.DATASET = self.dir / "dataset"
        for d in (C.TURNS, C.SCORES, C.EMBEDDINGS / "voiceclap", C.DATASET):
            d.mkdir(parents=True, exist_ok=True)
        self.write_inputs()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(C, k, v)
        self.tmp.cleanup()

    def write_inputs(self, drop=()):
        for stage in ("empathic", "voicenet", "voiceclap"):
            (C.SCORES / f"f.{stage}.jsonl").unlink(missing_ok=True)
        (C.EMBEDDINGS / "voiceclap" / "f.npy").unlink(missing_ok=True)
        (C.TURNS / "f.jsonl").write_text(
            "\n".join(json.dumps(t) for t in TURNS) + "\n", encoding="utf-8")
        if "empathic" not in drop:
            rows = [{"segment_id": t["segment_id"], "duration": t["duration"],
                     "empathic": {"Joy": 0.6, "Anger": 0.1},
                     "empathic_top5": [{"label": "Joy", "score": 0.6}]}
                    for t in TURNS]
            (C.SCORES / "f.empathic.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        if "voicenet" not in drop:
            rows = [{"segment_id": t["segment_id"], "duration": t["duration"],
                     "voicenet": {"reg": {"AROU": 3.0}, "cls": {"AROU": 2},
                                  "conf": {"AROU": 0.7}}}
                    for t in TURNS]
            (C.SCORES / "f.voicenet.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        if "voiceclap" not in drop:
            rows = [{"segment_id": t["segment_id"], "duration": t["duration"],
                     "voiceclap_top5": [{"label": "calm", "sim": 0.4}],
                     "voiceclap_best": "calm", "voiceclap_best_sim": 0.4}
                    for t in TURNS]
            (C.SCORES / "f.voiceclap.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            np.save(C.EMBEDDINGS / "voiceclap" / "f.npy",
                    np.ones((len(TURNS), 8), dtype=np.float16))

    def test_full_join(self):
        rows, err, delta = A.collect_rows("f")
        self.assertIsNone(err)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["segment_id"], "f_00000")
        self.assertEqual(rows[1]["speaker"], "speaker_1")
        self.assertAlmostEqual(rows[0]["empathic"]["Joy"], 0.6, places=5)
        self.assertEqual(rows[1]["voicenet"]["cls"]["AROU"], 2)
        self.assertEqual(rows[0]["voiceclap_emb_row"], 0)
        self.assertEqual(rows[1]["voiceclap_emb_row"], 1)
        self.assertEqual(delta["files"], 1)
        self.assertEqual(delta["turns"], 2)
        self.assertAlmostEqual(delta["speech_s"], 2.5, places=5)
        self.assertEqual(delta["with_empathic"], 2)
        self.assertEqual(delta["with_voiceclap"], 2)
        self.assertEqual(delta["empathic_top1"]["Joy"], 2)
        self.assertEqual(delta["speakers"], {"speaker_0", "speaker_1"})

    def test_missing_scores_are_null(self):
        self.write_inputs(drop=("empathic", "voiceclap"))
        rows, err, delta = A.collect_rows("f")
        self.assertIsNone(err)
        self.assertIsNone(rows[0]["empathic"])
        self.assertIsNone(rows[0]["voiceclap_top5"])
        self.assertIsNone(rows[0]["voiceclap_emb_row"])
        self.assertEqual(delta["with_empathic"], 0)
        self.assertEqual(delta["with_voiceclap"], 0)

    def test_missing_embedding_sidecar(self):
        (C.EMBEDDINGS / "voiceclap" / "f.npy").unlink()
        rows, _, _ = A.collect_rows("f")
        self.assertIsNone(rows[0]["voiceclap_emb_file"])
        self.assertIsNone(rows[0]["voiceclap_emb_row"])

    def test_no_turns(self):
        rows, err, delta = A.collect_rows("nope")
        self.assertIsNone(rows)
        self.assertEqual(err, "no_turns")
        self.assertIsNone(delta)

    def test_emb_row_counts_only_scored_turns(self):
        # Drop the voiceclap score of the first turn: the second turn becomes row 0.
        rows = [{"segment_id": t["segment_id"], "duration": t["duration"],
                 "voiceclap_top5": [{"label": "calm", "sim": 0.4}],
                 "voiceclap_best": "calm", "voiceclap_best_sim": 0.4}
                for t in TURNS[1:]]
        (C.SCORES / "f.voiceclap.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        out, _, delta = A.collect_rows("f")
        self.assertIsNone(out[0]["voiceclap_emb_row"])
        self.assertEqual(out[1]["voiceclap_emb_row"], 0)
        self.assertEqual(delta["with_voiceclap"], 1)


class TestMergeStats(unittest.TestCase):
    def test_merge(self):
        stats = {"files": 0, "turns": 0, "speech_s": 0.0, "with_empathic": 0,
                 "with_voicenet": 0, "with_voiceclap": 0,
                 "empathic_top1": Counter(),
                 "voiceclap_top1": Counter(),
                 "speakers": set()}
        d1 = {"files": 1, "turns": 2, "speech_s": 1.0, "with_empathic": 2,
              "with_voicenet": 1, "with_voiceclap": 2,
              "empathic_top1": Counter({"Joy": 2}),
              "voiceclap_top1": Counter({"calm": 2}),
              "speakers": {"speaker_0"}}
        A.merge_stats(stats, d1)
        A.merge_stats(stats, d1)
        A.merge_stats(stats, None)
        self.assertEqual(stats["files"], 2)
        self.assertEqual(stats["turns"], 4)
        self.assertAlmostEqual(stats["speech_s"], 2.0)
        self.assertEqual(stats["empathic_top1"]["Joy"], 4)
        self.assertEqual(stats["speakers"], {"speaker_0"})


if __name__ == "__main__":
    unittest.main()
