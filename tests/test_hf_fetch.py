"""Offline unit tests for the resumable ranged downloader (no network).

``Downloader._fetch_block`` is replaced by a fake that fills the requested byte
range with a deterministic pattern, so the block accounting, checkpoint format
and resume logic can be exercised without touching Hugging Face.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_fetch  # noqa: E402


def pattern(start: int, end: int) -> bytes:
    return bytes((i * 7 + 3) % 251 for i in range(start, end))


class FakeDownloader(hf_fetch.Downloader):
    """Records every requested block instead of issuing HTTP requests."""

    def __init__(self, *a, fail_blocks=(), **kw):
        super().__init__(*a, **kw)
        self.requests = []
        self.fail_blocks = set(fail_blocks)

    def _fetch_block(self, repo, path, part, start, end, repo_type="model"):
        self.requests.append((start, end))
        # Block index is derivable from the offset because blocks are uniform.
        i = start // self.block
        if i in self.fail_blocks:
            return False
        with open(part, "r+b") as fh:
            fh.seek(start)
            fh.write(pattern(start, end + 1))
        return True


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.size = 40  # tiny file, 4 blocks of 10 bytes

    def tearDown(self):
        self.tmp.cleanup()

    def dl(self, **kw):
        return FakeDownloader(token="x", block_mb=10 / (1024 * 1024), **kw)

    def test_full_download(self):
        d = self.dl()
        dest = self.dir / "f.bin"
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertTrue(ok, msg)
        self.assertEqual(dest.read_bytes(), pattern(0, self.size))
        self.assertEqual(sorted(d.requests), [(0, 9), (10, 19), (20, 29), (30, 39)])
        self.assertFalse(dest.with_name(dest.name + ".part").exists())
        self.assertFalse(dest.with_name(dest.name + ".blocks.json").exists())

    def test_present_file_is_skipped(self):
        dest = self.dir / "f.bin"
        dest.write_bytes(pattern(0, self.size))
        d = self.dl()
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertTrue(ok)
        self.assertEqual(msg, "present")
        self.assertEqual(d.requests, [])

    def test_resume_only_fetches_missing_blocks(self):
        dest = self.dir / "f.bin"
        part = dest.with_name(dest.name + ".part")
        state = dest.with_name(dest.name + ".blocks.json")
        block = 10
        # Simulate an interrupted run: 2 of 4 blocks done, part preallocated.
        part.write_bytes(b"\0" * self.size)
        with open(part, "r+b") as fh:
            fh.seek(0)
            fh.write(pattern(0, 20))
        state.write_text(json.dumps({"block": block, "done": [0, 1]}))

        d = FakeDownloader(token="x", block_mb=10 / (1024 * 1024))
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertTrue(ok, msg)
        self.assertEqual(sorted(d.requests), [(20, 29), (30, 39)])
        self.assertEqual(dest.read_bytes(), pattern(0, self.size))

    def test_block_size_change_discards_state(self):
        dest = self.dir / "f.bin"
        part = dest.with_name(dest.name + ".part")
        state = dest.with_name(dest.name + ".blocks.json")
        part.write_bytes(b"\0" * self.size)
        # State written with a different block size must not be trusted.
        state.write_text(json.dumps({"block": 8, "done": [0]}))

        d = FakeDownloader(token="x", block_mb=10 / (1024 * 1024))
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertTrue(ok, msg)
        self.assertEqual(len(d.requests), 4)
        self.assertEqual(dest.read_bytes(), pattern(0, self.size))

    def test_failed_block_keeps_partial_state(self):
        d = self.dl(fail_blocks={2})
        dest = self.dir / "f.bin"
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertFalse(ok)
        self.assertIn("failed", msg)
        self.assertFalse(dest.exists())
        state = dest.with_name(dest.name + ".blocks.json")
        self.assertEqual(json.loads(state.read_text())["done"], [0, 1, 3])

    def test_seeded_prefix_marks_covered_blocks(self):
        dest = self.dir / "f.bin"
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(pattern(0, 20) + b"\0" * 20)
        d = self.dl()
        ok, msg = d.download("r", "p", dest, self.size, assume_prefix=20)
        self.assertTrue(ok, msg)
        self.assertEqual(sorted(d.requests), [(20, 29), (30, 39)])

    def test_url_layout(self):
        d = self.dl()
        self.assertEqual(
            d._url("org/repo", "a/b.bin", "model"),
            "https://huggingface.co/org/repo/resolve/main/a/b.bin?download=true")
        self.assertEqual(
            d._url("org/ds", "x.wav", "dataset"),
            "https://huggingface.co/datasets/org/ds/resolve/main/x.wav?download=true")


if __name__ == "__main__":
    unittest.main()
