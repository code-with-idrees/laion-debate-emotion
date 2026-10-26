"""dl_progress.py - how much of each model repo is on disk."""
from __future__ import annotations

import fnmatch
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from huggingface_hub import HfApi

GROUPS = {
    "voicenet": (C.VOICENET_ID, C.VOICENET_DIR, None),
    "bude": (C.BUDE_ID, C.BUDE_DIR, None),
    "empathic": (C.EMPATHIC_PLUS_ID, C.EMPATHIC_PLUS_DIR,
                 ["model_*.pth", "*.json", "README.md"]),
    "voiceclap": (C.VOICECLAP_ID, C.VOICECLAP_DIR, None),
}


def main():
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    for key, (repo, local, patterns) in GROUPS.items():
        try:
            info = api.model_info(repo, files_metadata=True,
                                  token=os.environ.get("HF_TOKEN"))
        except Exception as e:
            print(f"{key:10s} ERR {type(e).__name__}: {e}")
            continue
        tot = have = 0
        missing = []
        for s in info.siblings:
            if patterns and not any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
                continue
            size = int(s.size or 0)
            tot += size
            dest = local / s.rfilename
            if dest.exists() and dest.stat().st_size == size:
                have += size
            else:
                missing.append(s.rfilename)
        part = 0
        cache = local / ".cache" / "huggingface" / "download"
        if cache.exists():
            part = sum(f.stat().st_size for f in cache.rglob("*.incomplete"))
        pct = 100 * have / tot if tot else 100
        print(f"{key:10s} {have/1e9:7.2f}/{tot/1e9:6.2f} GB {pct:5.1f}%  "
              f"missing={len(missing):3d}  in-flight={part/1e6:7.1f} MB")
        for m in missing[:3]:
            print(f"           - {m}")


if __name__ == "__main__":
    main()
