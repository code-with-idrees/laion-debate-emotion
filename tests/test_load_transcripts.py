"""Offline tests for 03_load_transcripts (corpus transcripts instead of ASR)."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_stage():
    spec = importlib.util.spec_from_file_location(
        "_script_03", ROOT / "scripts" / "03_load_transcripts.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load_stage()

RAW = [
    {"word": "It", "start": 9.28, "end": 9.6},
    {"word": "gives", "start": 9.6, "end": 10.08},
    {"word": "me", "start": 10.08, "end": 10.24},
    {"word": "great", "start": 10.64, "end": 11.28},
    {"word": "delight", "start": 11.28, "end": 11.28},
    {"word": " ", "start": 11.3, "end": 11.3},
]


class TestNormalise(unittest.TestCase):
    def test_leading_space_only_after_first_word(self):
        words = T.normalise_words(RAW)
        self.assertEqual(words[0]["word"], "It")
        self.assertTrue(all(w["word"].startswith(" ") for w in words[1:]))

    def test_drops_blank_and_untimed_words(self):
        words = T.normalise_words(RAW)
        self.assertEqual(len(words), 5)  # the " " placeholder is dropped
        for w in words:
            self.assertTrue(w["word"].strip())

    def test_sorts_by_start(self):
        words = T.normalise_words([{"word": "b", "start": 2.0, "end": 2.5},
                                   {"word": "a", "start": 1.0, "end": 1.5}])
        self.assertEqual([w["word"].strip() for w in words], ["a", "b"])

    def test_fills_missing_boundary(self):
        words = T.normalise_words([{"word": "a", "start": None, "end": 3.0},
                                   {"word": "b", "start": 4.0, "end": None}])
        self.assertEqual((words[0]["start"], words[0]["end"]), (3.0, 3.0))
        self.assertEqual((words[1]["start"], words[1]["end"]), (4.0, 4.0))

    def test_repairs_reversed_times(self):
        w = T.normalise_words([{"word": "a", "start": 5.0, "end": 4.0}])[0]
        self.assertEqual((w["start"], w["end"]), (4.0, 5.0))

    def test_empty_input(self):
        self.assertEqual(T.normalise_words(None), [])
        self.assertEqual(T.normalise_words([]), [])


class TestSegments(unittest.TestCase):
    def test_gap_starts_new_segment(self):
        words = T.normalise_words(RAW)
        segs = T.to_segments(words)
        # 10.24 -> 10.64 is a 0.40 s pause, exactly SEG_GAP_S: not a split.
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0]["text"], "It gives me great delight")

    def test_long_segment_is_split(self):
        words = [{"word": f" w{i}", "start": i * 1.0, "end": i * 1.0 + 0.9,
                  "probability": None} for i in range(20)]
        segs = T.to_segments(words)
        self.assertGreater(len(segs), 1)
        for s in segs:
            self.assertLessEqual(s["end"] - s["start"], T.SEG_MAX_S + 1.0)

    def test_segment_text_joins_without_double_spaces(self):
        words = T.normalise_words(RAW)
        segs = T.to_segments(words)
        self.assertNotIn("  ", segs[0]["text"])
        self.assertEqual(segs[0]["text"], segs[0]["text"].strip())

    def test_pause_beyond_gap_splits(self):
        words = T.normalise_words(RAW + [{"word": "later", "start": 20.0,
                                          "end": 20.5}])
        self.assertEqual(len(T.to_segments(words)), 2)

    def test_exact_gap_does_not_split(self):
        # 10.64 - 10.24 == SEG_GAP_S exactly; floating point must not decide.
        words = T.normalise_words(RAW)
        self.assertEqual(len(T.to_segments(words)), 1)


class TestRecord(unittest.TestCase):
    def test_shape_matches_whisper_output(self):
        rec = {"filename": "x.wav", "transcript": "ignored",
               "word_timestamps": RAW}
        out = T.build_record(rec, "x", "D:/audio/x.wav")
        for key in ("file_id", "audio", "duration", "language", "segments"):
            self.assertIn(key, out)
        self.assertEqual(out["source"], "dataset_transcript")
        self.assertAlmostEqual(out["duration"], 11.28, places=3)
        seg = out["segments"][0]
        for key in ("id", "start", "end", "text", "words"):
            self.assertIn(key, seg)
        self.assertIsNone(out["language_probability"])

    def test_empty_transcript(self):
        out = T.build_record({"word_timestamps": []}, "x", "")
        self.assertEqual(out["segments"], [])
        self.assertEqual(out["duration"], 0.0)


if __name__ == "__main__":
    unittest.main()
