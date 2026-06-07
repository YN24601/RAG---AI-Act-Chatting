"""Day 1-2 ingestion entrypoint: fetch -> parse -> chunk.

Usage:
    python scripts/run_ingestion.py                 # full pipeline
    python scripts/run_ingestion.py --force         # re-download HTML
    python scripts/run_ingestion.py --steps parse,chunk
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingestion import chunk as chunk_mod  # noqa: E402
from ingestion import parse as parse_mod  # noqa: E402
from ingestion.fetch import fetch  # noqa: E402


def _print_summary(stats: dict) -> None:
    print("\n=== chunking comparison (tiktoken cl100k tokens) ===")
    header = f"{'strategy':<10} {'chunks':>7} {'mean':>7} {'median':>7} {'p95':>6}"
    print(header)
    print("-" * len(header))
    for name in ("baseline", "structure"):
        s = stats[name]
        print(f"{name:<10} {s['count']:>7} {s['mean']:>7} {s['median']:>7} {s['p95']:>6}")


def main() -> None:
    ap = argparse.ArgumentParser(description="EU AI Act ingestion (Day 1-2)")
    ap.add_argument("--force", action="store_true", help="re-download the HTML snapshot")
    ap.add_argument(
        "--steps",
        default="fetch,parse,chunk",
        # default="chunk",
        help="comma-separated subset of: fetch,parse,chunk (default: all)",
    )
    args = ap.parse_args()
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    html = None
    units = None

    # raw/
    if "fetch" in steps:
        html = fetch(force=args.force)

    # units.jsonl
    if "parse" in steps:
        if html is None:
            from ingestion.schema import RAW_HTML_PATH

            html = RAW_HTML_PATH.read_text(encoding="utf-8")
        units = parse_mod.parse(html)
        parse_mod.write_units(units)

    # chunks_*.jsonl
    if "chunk" in steps:
        if units is None:
            units = parse_mod.load_units()
        stats = chunk_mod.build_and_write(units)
        _print_summary(stats)

    print("\n[done] ingestion complete.")


if __name__ == "__main__":
    main()
