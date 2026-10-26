"""01 - Download the gated sleeping-debate WAVs (resumable, parallel, robust).

Uses the ranged block downloader (hf_fetch.py) instead of hf_hub_download so a
436 GB pull survives stalls: every 16 MB block is checkpointed, retried with a
socket timeout, and resumes exactly where it stopped.

Examples
--------
  python 01_download_dataset.py --dry-run
  python 01_download_dataset.py --workers 4 --chunk-threads 4
  python 01_download_dataset.py --limit 20
  python 01_download_dataset.py --ids Q2VANpuVNds POG4_VPK0V0
"""
from __future__ import annotations

import argparse
import csv
import json
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
    files = []
    for s in info.siblings:
        if s.rfilename.lower().endswith(".wav"):
            sha = getattr(s.lfs, "sha256", None) if s.lfs is not None else None
            files.append({"file_id": Path(s.rfilename).stem,
                          "filename": s.rfilename,
                          "size": int(s.size or 0),
                          "sha256": sha})
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


def check_access(dl: Downloader, filename: str) -> None:
    """One-byte ranged GET: fails loudly when the token is not authorized."""
    r = dl._session().get(
        dl._url(C.DATASET_REPO, filename, "dataset"),
        headers={"Range": "bytes=0-0"}, stream=True, timeout=(15, 30))
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
    ap.add_argument("--workers", type=int, default=4, help="files in parallel")
    ap.add_argument("--chunk-threads", type=int, default=4, help="connections per file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retries", type=int, default=8)
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
          f"size={total_gb:.2f} GB")
    if args.dry_run:
        for f in files[:10]:
            print(f"  {f['filename']}  {f['size']/1e6:.1f} MB")
        if len(files) > 10:
            print(f"  ... {len(files)-10} more")
        return
    if not files:
        print("no files selected")
        return

    dl = Downloader(os.environ.get("HF_TOKEN"),
                    chunk_threads=args.chunk_threads, retries=args.retries)
    check_access(dl, files[0]["filename"])

    manifest_path = C.MANIFESTS / "files.csv"
    results = []
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            results = list(csv.DictReader(fh))
    done_ids = {r["file_id"] for r in results if r.get("status") == "ok"}
    print(f"already ok: {len(done_ids)}")

    lock = threading.Lock()
    t0 = time.time()
    counter = {"ok": 0, "skip": 0, "fail": 0, "bytes": 0}
    stop = threading.Event()

    def monitor():
        last = dl.done_bytes
        while not stop.wait(30):
            with lock:
                n = counter["ok"] + counter["fail"]
                rate = dl.done_bytes / max(time.time() - t0, 1e-6) / 1e6
            print(f"    ... {n}/{len(files)} files, "
                  f"{dl.done_bytes/1e9:.2f} GB this run ({rate:.2f} MB/s)",
                  flush=True)
            last = dl.done_bytes

    th = threading.Thread(target=monitor, daemon=True)
    th.start()

    def one(f):
        dest = C.AUDIO / f["filename"]
        if not args.overwrite and f["file_id"] in done_ids:
            with lock:
                counter["skip"] += 1
            return {"file_id": f["file_id"], **f, "path": str(dest),
                    "status": "skip"}
        ts = time.time()
        good, msg = dl.download(C.DATASET_REPO, f["filename"], dest,
                                f["size"], repo_type="dataset")
        row = {"file_id": f["file_id"], **f, "path": str(dest),
               "seconds": round(time.time() - ts, 2)}
        if good:
            meta = probe(dest)
            row.update(meta)
            row["status"] = "ok" if "error" not in meta else "error"
            if "error" in meta:
                row["duration_s"] = ""
            with lock:
                counter["ok" if row["status"] == "ok" else "fail"] += 1
                counter["bytes"] += f["size"]
                n = counter["ok"]
                rate = counter["bytes"] / max(time.time() - t0, 1e-6) / 1e6
                print(f"[{n}/{len(files)}] {f['file_id']} {msg} "
                      f"({rate:.2f} MB/s agg)", flush=True)
        else:
            row["status"] = "error"
            row["error"] = msg
            with lock:
                counter["fail"] += 1
            print(f"FAIL {f['file_id']}: {msg}", flush=True)
        return row

    pending = [f for f in files
               if args.overwrite or f["file_id"] not in done_ids]
    if pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(one, f) for f in pending]
            for k, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                with lock:
                    results = [x for x in results if x["file_id"] != r["file_id"]]
                    results.append(r)
                if k % 5 == 0:
                    write_manifest(manifest_path, results)
    stop.set()
    write_manifest(manifest_path, results)
    print(f"done ok={counter['ok']} skip={counter['skip']} fail={counter['fail']} "
          f"in {(time.time()-t0)/60:.1f} min")
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
