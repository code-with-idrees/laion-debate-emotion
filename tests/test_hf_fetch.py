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

    def test_truncated_state_is_salvaged(self):
        """A kill mid-checkpoint must not throw away finished blocks."""
        dest = self.dir / "f.bin"
        part = dest.with_name(dest.name + ".part")
        state = dest.with_name(dest.name + ".blocks.json")
        # Blocks 0 and 1 were written and recorded; the JSON was cut while
        # block 2's index was being appended.
        part.write_bytes(pattern(0, 20) + b"\0" * 20)
        state.write_text('{"block": 10, "done": [0, 1, 2')

        d = FakeDownloader(token="x", block_mb=10 / (1024 * 1024))
        ok, msg = d.download("r", "p", dest, self.size)
        self.assertTrue(ok, msg)
        # Only blocks 2 and 3 are fetched again.
        self.assertEqual(sorted(d.requests), [(20, 29), (30, 39)])

    def test_state_write_is_atomic(self):
        state = self.dir / "f.blocks.json"
        d = self.dl()
        d._save_state(state, {0, 1})
        self.assertEqual(json.loads(state.read_text())["done"], [0, 1])
        self.assertFalse(state.with_name(state.name + ".tmp").exists())

    def test_url_layout(self):
        d = self.dl()
        self.assertEqual(
            d._url("org/repo", "a/b.bin", "model"),
            "https://huggingface.co/org/repo/resolve/main/a/b.bin?download=true")
        self.assertEqual(
            d._url("org/ds", "x.wav", "dataset"),
            "https://huggingface.co/datasets/org/ds/resolve/main/x.wav?download=true")


class FakeResponse:
    """Just enough of a requests response for _read_block."""

    def __init__(self, pieces):
        self.pieces = pieces
        self.closed = False

    def iter_content(self, chunk_size=0):
        for p in self.pieces:
            yield p

    def close(self):
        self.closed = True


class TestReadBlock(unittest.TestCase):
    def test_reads_until_want(self):
        d = hf_fetch.Downloader(token="x")
        r = FakeResponse([b"a" * 10, b"b" * 6, b"c" * 100])
        self.assertEqual(d._read_block(r, 16), b"a" * 10 + b"b" * 6)

    def test_short_body_is_returned_as_is(self):
        d = hf_fetch.Downloader(token="x")
        self.assertEqual(d._read_block(FakeResponse([b"a" * 4]), 16), b"a" * 4)

    def test_trickling_block_trips_the_deadline(self):
        """A byte at a time never trips the socket timeout, only the deadline."""
        d = hf_fetch.Downloader(token="x", block_deadline=0)
        with self.assertRaises(RuntimeError):
            d._read_block(FakeResponse([b"a"] * 8), 16)


if __name__ == "__main__":
    unittest.main()
