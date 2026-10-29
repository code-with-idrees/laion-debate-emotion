"""Offline tests for the progress dashboard helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import progress  # noqa: E402


class TestBar(unittest.TestCase):
    def test_width_is_fixed(self):
        for frac in (0.0, 0.5, 1.0, 2.0, -1.0):
            self.assertEqual(len(progress.bar(frac, 10)), 10)

    def test_clamped(self):
        self.assertEqual(progress.bar(2.0, 4), "####")
        self.assertEqual(progress.bar(-1.0, 4), "....")


class TestHuman(unittest.TestCase):
    def test_hours_minutes(self):
        self.assertEqual(progress.human(3661), "01h 01m")

    def test_days(self):
        self.assertEqual(progress.human(90061), "1d 01h 01m")

    def test_unknown(self):
        self.assertEqual(progress.human(float("nan")), "?")
        self.assertEqual(progress.human(0), "?")


class TestDoneBytes(unittest.TestCase):
    def test_counts_wav_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "a.wav").write_bytes(b"\0" * 1000)
            (d / "b.wav.part").write_bytes(b"\0" * 999999)  # preallocated
            (d / "b.wav.blocks.json").write_text(json.dumps(
                {"block": 400, "done": [0, 1]}))
            self.assertEqual(progress.done_bytes(d), 1000 + 800)

    def test_last_block_is_clamped_to_file_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "b.wav.part").write_bytes(b"\0" * 950)  # 400 + 400 + 150
            (d / "b.wav.blocks.json").write_text(json.dumps(
                {"block": 400, "done": [0, 1, 2]}))
            # Not 3 * 400: the file is only 950 bytes long.
            self.assertEqual(progress.done_bytes(d), 950)

    def test_completed_file_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "b.wav").write_bytes(b"\0" * 800)
            (d / "b.wav.blocks.json").write_text(json.dumps(
                {"block": 400, "done": [0, 1]}))
            self.assertEqual(progress.done_bytes(d), 800)

    def test_ignores_corrupt_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "c.wav.blocks.json").write_text("{not json")
            (d / "c.wav.part").write_bytes(b"\0" * 400)
            self.assertEqual(progress.done_bytes(d), 0)


class TestManifest(unittest.TestCase):
    def test_counts_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "files.csv"
            p.write_text("file_id,status\na,ok\nb,error\nc,ok\n", encoding="utf-8")
            c = progress.manifest(p)
        self.assertEqual((c["ok"], c["error"], c["total"]), (2, 1, 3))


class TestLogIdle(unittest.TestCase):
    def _log(self, text: str) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False,
                                          encoding="utf-8")
        tmp.write(text)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_reads_last_idle_value(self):
        p = self._log("[x] ... 4/2312 files, 0.49 GB done (0.52 GB fetched, "
                      "2.89 MB/s avg)\n"
                      "[x] ... 4/2312 files, 0.49 GB done (0.53 GB fetched, "
                      "2.10 MB/s avg, idle 7 min)\n")
        self.assertEqual(progress.log_idle_minutes(p), 7.0)

    def test_none_when_no_idle_yet(self):
        p = self._log("[x] ... 4/2312 files, 0.49 GB done (0.52 GB fetched, "
                      "2.89 MB/s avg)\n")
        self.assertIsNone(progress.log_idle_minutes(p))

    def test_missing_log(self):
        self.assertIsNone(progress.log_idle_minutes(Path("nope.log")))


if __name__ == "__main__":
    unittest.main()
