"""Build Qdrant collections from the chunk sets (Day 3-4).

Usage:
    python scripts/build_index.py                 # index baseline + structure
    python scripts/build_index.py --recreate      # force rebuild
    python scripts/build_index.py --strategy structure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.index import build_all  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Index chunks into Qdrant (Day 3-4)")
    ap.add_argument(
        "--strategy",
        choices=["baseline", "structure", "both"],
        default="both",
        help="which chunk set(s) to index (default: both)",
    )
    ap.add_argument("--recreate", action="store_true", help="rebuild even if counts match")
    args = ap.parse_args()

    strategies = ("baseline", "structure") if args.strategy == "both" else (args.strategy,)
    build_all(strategies=strategies, recreate=args.recreate)
    print("\n[done] indexing complete.")


if __name__ == "__main__":
    main()
