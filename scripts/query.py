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
    ap.add_argument("--number-min", type=int, default=None, help="min article/recital number")
    ap.add_argument("--number-max", type=int, default=None, help="max article/recital number")
    ap.add_argument("--min-score", type=float, default=None, help="drop hits below this score")
    ap.add_argument(
        "--rerank",
        choices=["default", "off", "cohere"],
        default="default",
        help="second-stage reranking override (default uses RERANK_MODE)",
    )
    args = ap.parse_args()

    query = " ".join(args.query)
    retriever = Retriever(strategy=args.strategy, k=args.k)
    rerank_override = None if args.rerank == "default" else args.rerank == "cohere"
    outcome = retriever.search_with_outcome(
        query,
        k=args.k,
        top_n=args.top_n,
        unit_type=args.unit_type,
        number_min=args.number_min,
        number_max=args.number_max,
        min_score=args.min_score,
        rerank=rerank_override,
    )
    hits = outcome.hits

    print(f"\nquery     : {query}")
    print(f"collection: {args.strategy}  (recall k={args.k}, showing top {args.top_n})")
    print(f"rerank    : {outcome.status}  model={outcome.model or 'none'}  latency={outcome.latency_ms:.2f}ms")
    filters = []
    if args.unit_type:
        filters.append(f"unit_type=={args.unit_type}")
    if args.number_min is not None or args.number_max is not None:
        filters.append(f"number_int in [{args.number_min}, {args.number_max}]")
    if args.min_score is not None:
        filters.append(f"score>={args.min_score}")
    if filters:
        print(f"filter    : {'  '.join(filters)}")
    print("=" * 78)
    if not hits:
        print("\n(no results above the given constraints)\n")
        return
    for h in hits:
        m = h.metadata
        loc = m.get("context_header") or f"chunk {m.get('chunk_index')}"
        chapter = f"  [{m['chapter']}]" if m.get("chapter") else ""
        rerank_score = f"  rerank_score={h.rerank_score:.4f}" if h.rerank_score is not None else ""
        print(
            f"\n#{h.rank}  vector_rank={h.retrieval_rank}  "
            f"vector_score={h.score}{rerank_score}  ({h.chunk_id})"
        )
        print(f"    {loc}{chapter}")
        snippet = " ".join(h.text.split())[:280]
        print(f"    {snippet}…")
    print()


if __name__ == "__main__":
    main()
