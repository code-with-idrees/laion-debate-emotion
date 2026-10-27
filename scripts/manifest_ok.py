"""Print how many manifest rows are usable (status ok or skip).

Used by the download supervisor to decide whether another pass is needed, and
handy on its own: ``python scripts/manifest_ok.py``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


def counts(path: Path | None = None) -> dict[str, int]:
    path = path or (C.MANIFESTS / "files.csv")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out: dict[str, int] = {}
    for r in rows:
        key = r.get("status") or "unknown"
        out[key] = out.get(key, 0) + 1
    out["total"] = len(rows)
    return out


def main() -> int:
    c = counts()
    print(c.get("ok", 0))
    if len(sys.argv) > 1 and sys.argv[1] == "--verbose":
        print(c, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
