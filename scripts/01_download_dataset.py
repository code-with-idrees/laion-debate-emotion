"""01 - Download the gated sleeping-debate WAVs (resumable, parallel).

436 GB over a throttled link. Throughput scales with the number of *files* in
flight: one stream manages ~1.5 MB/s, four ~1.8 MB/s, and the aggregate keeps
creeping up while per-stream speed drops. HF's Xet transfer is faster when it
works (measured 3 MB/s) but intermittently stalls at zero bytes for minutes, so
it is opt-in via ``--engine xet``.

The default engine is the ranged downloader from ``hf_fetch.py``: every file is
split into 16 MB blocks, each block is a timed Range request, finished blocks
are checkpointed in ``<name>.blocks.json`` and a stall only costs one block.

Examples
--------
  python 01_download_dataset.py --dry-run
  python 01_download_dataset.py --workers 8
  python 01_download_dataset.py --limit 20
  python 01_download_dataset.py --ids Q2VANpuVNds POG4_VPK0V0
  python 01_download_dataset.py --engine xet
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from hf_fetch import Downloader

import soundfile as sf
from huggingface_hub import HfApi

FIELDS = ["file_id", "filename", "size", "duration_s", "sample_rate",
          "channels", "frames", "subtype", "path", "status", "seconds"]


def list_files(api: HfApi):
    info = api.dataset_info(C.DATASET_REPO, files_metadata=True,
                            token=os.environ.get("HF_TOKEN"))
    files = [{"file_id": Path(s.rfilename).stem,
              "filename": s.rfilename,
              "size": int(s.size or 0)}
             for s in info.siblings if s.rfilename.lower().endswith(".wav")]
    files.sort(key=lambda x: x["filename"])
    return files, info


def probe(path: Path):
    try:
        info = sf.info(str(path))
        return {
            "duration_s": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "subtype": info.subtype,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def check_access(filename: str) -> None:
    """One-byte ranged GET: fails loudly when the token is not authorized."""
    import requests

    url = (f"https://huggingface.co/datasets/{C.DATASET_REPO}"
           f"/resolve/main/{filename}")
    r = requests.get(url, headers={"Authorization":
                                   f"Bearer {os.environ.get('HF_TOKEN', '')}",
                                   "Range": "bytes=0-0"},
                     timeout=(15, 30), stream=True)
    r.close()
    if r.status_code == 403:
        print("\nACCESS DENIED - the token is not on the authorized list.")
        print("Ask the dataset owner to approve your request at:")
        print(f"  https://huggingface.co/datasets/{C.DATASET_REPO}")
        raise SystemExit(2)
    if r.status_code not in (200, 206):
        raise SystemExit(f"unexpected HTTP {r.status_code} for {filename}")


def write_manifest(path: Path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["file_id"]):
            w.writerow(r)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="files in parallel")
    ap.add_argument("--chunk-threads", type=int, default=1,
                    help="parallel 16 MB blocks per file (ranged engine)")
    ap.add_argument("--engine", choices=("ranged", "xet"), default="ranged")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    files, info = list_files(api)
    if args.ids:
        wanted = set(args.ids)
        files = [f for f in files if f["file_id"] in wanted]
    if args.limit:
        files = files[: args.limit]

    total_gb = sum(f["size"] for f in files) / 1e9
    print(f"repo={C.DATASET_REPO} gated={info.gated} files={len(files)} "
          f"size={total_gb:.2f} GB engine={args.engine} workers={args.workers}",
          flush=True)
    if args.dry_run:
        for f in files[:10]:
            print(f"  {f['filename']}  {f['size'] / 1e6:.1f} MB")
        if len(files) > 10:
            print(f"  ... {len(files) - 10} more")
        return
    if not files:
        print("no files selected")
        return

    check_access(files[0]["filename"])

    manifest_path = C.MANIFESTS / "files.csv"
    results = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            results = list(csv.DictReader(fh))
    done_ids = {r["file_id"] for r in results if r.get("status") == "ok"}

    pending = [f for f in files
               if args.overwrite or f["file_id"] not in done_ids]
    if not pending:
        print(f"nothing to download ({len(files)} already ok)", flush=True)
        return
    print(f"already ok: {len(done_ids)}; queued: {len(pending)}", flush=True)

    dl = None
    if args.engine == "xet":
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
        from huggingface_hub import hf_hub_download

        def fetch(f):
            path = hf_hub_download(C.DATASET_REPO, f["filename"],
                                   repo_type="dataset", local_dir=str(C.AUDIO),
                                   token=os.environ.get("HF_TOKEN"))
            size = Path(path).stat().st_size
            if f["size"] and size != f["size"]:
                raise RuntimeError(f"size {size} != {f['size']}")
    else:
        dl = Downloader(os.environ.get("HF_TOKEN"),
                        chunk_threads=args.chunk_threads)

        def fetch(f):
            dest = C.AUDIO / f["filename"]
            ok, msg = dl.download(C.DATASET_REPO, f["filename"], dest,
                                  f["size"], repo_type="dataset")
            if not ok:
                raise RuntimeError(msg)

    lock = threading.Lock()
    t0 = time.time()
    counter = {"ok": 0, "fail": 0, "bytes": 0}
    stop = threading.Event()

    def monitor():
        while not stop.wait(30):
            with lock:
                done = counter["ok"] + counter["fail"]
                gb = counter["bytes"] / 1e9
            # Files only count when they finish; in-flight block progress keeps
            # the rate honest for 150 MB recordings.
            live = gb + ((dl.done_bytes if dl else 0) / 1e9)
            rate = live * 1e9 / max(time.time() - t0, 1e-6) / 1e6
            print(f"    ... {done}/{len(pending)} files, {gb:.2f} GB done "
                  f"({live:.2f} GB in flight, {rate:.2f} MB/s)", flush=True)

    def one(f):
        ts = time.time()
        row = {"file_id": f["file_id"], **f,
               "path": str(C.AUDIO / f["filename"])}
        try:
            fetch(f)
            row.update(probe(Path(row["path"])))
            row["status"] = "ok" if "error" not in row else "error"
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        row["seconds"] = round(time.time() - ts, 2)
        with lock:
            counter["ok" if row["status"] == "ok" else "fail"] += 1
            if row["status"] == "ok":
                counter["bytes"] += f["size"]
            n = counter["ok"] + counter["fail"]
            rate = counter["bytes"] / max(time.time() - t0, 1e-6) / 1e6
        if row["status"] == "ok":
            print(f"[{n}/{len(pending)}] {f['file_id']} "
                  f"{f['size'] / 1e6:.1f} MB in {row['seconds']:.0f}s "
                  f"({rate:.2f} MB/s agg)", flush=True)
        else:
            print(f"FAIL {f['file_id']}: {row['error']}", flush=True)
        return row

    threading.Thread(target=monitor, daemon=True).start()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, f): f for f in pending}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            with lock:
                results = [x for x in results if x["file_id"] != r["file_id"]]
                results.append(r)
            if k % 5 == 0:
                write_manifest(manifest_path, results)
    stop.set()
    write_manifest(manifest_path, results)
    print(f"done ok={counter['ok']} fail={counter['fail']} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
