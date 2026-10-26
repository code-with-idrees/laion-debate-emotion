"""Offline unit tests for the word -> speaker-turn alignment (script 04).

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(stem: str):
    """Load ``scripts/<stem>.py`` by path (names start with a digit)."""
    spec = importlib.util.spec_from_file_location(
        "_script_" + stem, ROOT / "scripts" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load_script("04_build_turns")


class TestActivity(unittest.TestCase):
    def test_shape_and_frames(self):
        segs = [{"Start": 0.0, "End": 1.0, "Speaker": 0},
                {"Start": 1.0, "End": 2.0, "Speaker": 1}]
        ids, act, n = T.build_activity(segs, 2.0)
        self.assertEqual(ids, [0, 1])
        self.assertEqual(n, 200)          # 2 s / 10 ms
        self.assertEqual(act.shape, (2, 200))
        self.assertTrue(act[0, :100].all())
        self.assertFalse(act[0, 100:].any())
        self.assertTrue(act[1, 100:].all())

    def test_degenerate_segments_are_ignored(self):
        ids, act, n = T.build_activity(
            [{"Start": 1.0, "End": 1.0, "Speaker": 3}], 1.0)
        self.assertEqual(ids, [])
        self.assertEqual(act.shape, (0, 100))

    def test_segment_past_end_is_clipped(self):
        ids, act, n = T.build_activity(
            [{"Start": 0.5, "End": 99.0, "Speaker": 0}], 1.0)
        self.assertEqual(act.shape, (1, 100))
        self.assertTrue(act[0, 50:].all())


class TestWordSpeaker(unittest.TestCase):
    def setUp(self):
        segs = [{"Start": 0.0, "End": 1.0, "Speaker": 0},
                {"Start": 1.0, "End": 2.0, "Speaker": 1}]
        self.ids, self.act, _ = T.build_activity(segs, 2.0)

    def test_clean_assignment(self):
        spk, _ = T.word_speaker({"start": 0.1, "end": 0.4}, self.act, self.ids)
        self.assertEqual(spk, 0)
        spk, _ = T.word_speaker({"start": 1.5, "end": 1.8}, self.act, self.ids)
        self.assertEqual(spk, 1)

    def test_tie_broken_by_midpoint(self):
        # 0.8-1.2 overlaps each speaker by exactly 0.2 s; the midpoint is 1.0,
        # which is active for speaker 1 because activity is half-open.
        spk, _ = T.word_speaker({"start": 0.8, "end": 1.2}, self.act, self.ids)
        self.assertEqual(spk, 1)

    def test_silent_gap_is_unassigned(self):
        segs = [{"Start": 0.0, "End": 0.5, "Speaker": 0},
                {"Start": 2.5, "End": 3.0, "Speaker": 0}]
        ids, act, _ = T.build_activity(segs, 3.0)
        spk, ov = T.word_speaker({"start": 1.4, "end": 1.6}, act, ids)
        self.assertIsNone(spk)
        self.assertEqual(ov, 0.0)

    def test_beyond_duration_clamps_to_last_frame(self):
        # Whisper occasionally emits a word that ends a few ms past the audio;
        # frames are clamped so the word still gets the last active speaker.
        spk, _ = T.word_speaker({"start": 5.0, "end": 5.2}, self.act, self.ids)
        self.assertEqual(spk, 1)

    def test_missing_end_uses_start(self):
        spk, _ = T.word_speaker({"start": 1.5, "end": None}, self.act, self.ids)
        self.assertEqual(spk, 1)

    def test_no_speakers(self):
        spk, ov = T.word_speaker({"start": 0.0, "end": 1.0},
                                 self.act[:0], [])
        self.assertIsNone(spk)
        self.assertEqual(ov, 0.0)

    def test_overlap_flag(self):
        segs = [{"Start": 0.0, "End": 2.0, "Speaker": 0},
                {"Start": 0.0, "End": 1.0, "Speaker": 1}]
        ids, act, _ = T.build_activity(segs, 2.0)
        _, ov = T.word_speaker({"start": 0.0, "end": 1.0}, act, ids)
        self.assertAlmostEqual(ov, 1.0, places=6)


class TestGrouping(unittest.TestCase):
    @staticmethod
    def w(word, start, end, speaker):
        return {"word": word, "start": start, "end": end,
                "speaker": speaker, "overlap": 0.0, "probability": 0.9}

    def test_speaker_change_splits(self):
        words = [self.w("a", 0.0, 0.3, 0), self.w("b", 0.3, 0.6, 1)]
        self.assertEqual([len(g) for g in T.group_words(words)], [1, 1])

    def test_gap_splits(self):
        words = [self.w("a", 0.0, 0.3, 0), self.w("b", 1.5, 1.8, 0)]
        self.assertEqual([len(g) for g in T.group_words(words)], [1, 1])

    def test_short_gap_keeps_turn(self):
        words = [self.w("a", 0.0, 0.3, 0), self.w("b", 0.5, 0.8, 0)]
        self.assertEqual([len(g) for g in T.group_words(words)], [2])

    def test_long_turn_splits(self):
        limit = T.C.TURN_MAX_S
        words = [self.w("a", 0.0, 0.4, 0),
                 self.w("b", limit - 0.1, limit + 0.1, 0)]
        self.assertEqual([len(g) for g in T.group_words(words)], [1, 1])

    def test_unassigned_words_dropped(self):
        words = [self.w("a", 0.0, 0.3, None), self.w("b", 0.4, 0.7, 2)]
        groups = list(T.group_words(words))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][0]["word"], "b")


if __name__ == "__main__":
    unittest.main()
