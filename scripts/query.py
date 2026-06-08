"""Query a Qdrant collection and print ranked retrieval results (Day 3-4).

Usage:
    python scripts/query.py "What are prohibited AI practices?"
    python scripts/query.py "high-risk classification" --strategy baseline
    python scripts/query.py "definition of provider" --unit-type article --k 20 --top-n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.config import DEFAULT_K, DEFAULT_TOP_N  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve from Qdrant (Day 3-4)")
    ap.add_argument("query", nargs="+", help="the question / search query")
    ap.add_argument("--strategy", choices=["baseline", "structure"], default="structure")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="vector recall depth")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="results to show")
    ap.add_argument("--unit-type", choices=["recital", "article", "annex"], default=None)
    args = ap.parse_args()

    query = " ".join(args.query)
    retriever = Retriever(strategy=args.strategy, k=args.k)
    hits = retriever.search(query, k=args.k, top_n=args.top_n, unit_type=args.unit_type)

    print(f"\nquery     : {query}")
    print(f"collection: {args.strategy}  (recall k={args.k}, showing top {args.top_n})")
    if args.unit_type:
        print(f"filter    : unit_type == {args.unit_type}")
    print("=" * 78)
    for h in hits:
        m = h.metadata
        loc = m.get("context_header") or f"chunk {m.get('chunk_index')}"
        chapter = f"  [{m['chapter']}]" if m.get("chapter") else ""
        print(f"\n#{h.rank}  score={h.score}  ({h.chunk_id})")
        print(f"    {loc}{chapter}")
        snippet = " ".join(h.text.split())[:280]
        print(f"    {snippet}…")
    print()


if __name__ == "__main__":
    main()
