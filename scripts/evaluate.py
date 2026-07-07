"""Day 8-9 evaluation: RAGAS + refusal metrics + MLflow, per chunking strategy.

Runs the production RAG pipeline over the committed eval set for one or both
chunking strategies (the only eval axis this phase; rerank stays off), scores the
answered subset with RAGAS, scores the refusal decision on the full set, logs
everything to MLflow as a parent 'compare' run with nested per-strategy runs, and
optionally persists the eval set as a LangSmith dataset.

Usage:
    python scripts/evaluate.py --strategy all
    python scripts/evaluate.py --strategy structure --limit 3        # smoke test
    python scripts/evaluate.py --strategy all --langsmith-upload
    python scripts/evaluate.py --strategy structure --no-ragas       # refusal+latency only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.aggregate import latency_summary, partition_for_ragas  # noqa: E402
from evaluation.harness import run_over_set  # noqa: E402
from evaluation.refusal import refusal_scores  # noqa: E402
from evaluation.schema import eval_set_hash, load_eval_set  # noqa: E402
from evaluation.tracking import build_comparison_table, log_mlflow, push_langsmith_dataset  # noqa: E402
from generation import config as gconfig  # noqa: E402
from retrieval import config as rconfig  # noqa: E402


def evaluate_strategy(items, strategy: str, run_ragas: bool, pause_s: float = 0.0) -> dict:
    print(f"\n=== strategy: {strategy} ===")
    results = run_over_set(items, strategy, pause_s=pause_s)

    answered, _ = partition_for_ragas(results)
    if run_ragas:
        from evaluation.ragas_eval import ragas_version, score_answered  # local: heavy dep
        print(f"  scoring {len(answered)}/{len(results)} answered items with RAGAS (Mistral judge)...")
        ragas = score_answered(answered)
        rv = ragas_version()
    else:
        ragas = {}
        rv = "skipped"

    refusal = refusal_scores(
        expected=[r.item.should_refuse for r in results],
        actual=[r.refused for r in results],
    )
    latency = latency_summary(results)

    params = {
        "strategy": strategy,
        "k": rconfig.DEFAULT_K,
        "top_n": gconfig.ANSWER_TOP_N,
        "rerank": "off",  # identity passthrough this phase (honest)
        "embed_model": rconfig.EMBED_MODEL,
        "gen_model": gconfig.GEN_MODEL,
        "grade_min_score": gconfig.GRADE_MIN_SCORE,
        "ragas_version": rv,
        "eval_set_hash": eval_set_hash(),
        "n_items": len(results),
    }
    return {"strategy": strategy, "params": params, "ragas": ragas,
            "refusal": refusal, "latency": latency, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description="EU AI Act RAG evaluation (Day 8-9)")
    ap.add_argument("--strategy", choices=["baseline", "structure", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS (refusal + latency only)")
    ap.add_argument("--no-mlflow", action="store_true", help="don't log to MLflow")
    ap.add_argument("--langsmith-upload", action="store_true", help="persist eval set as LangSmith dataset")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between items (smooths Mistral rate limit)")
    ap.add_argument("--experiment", default="aiact-rag-eval")
    args = ap.parse_args()

    items = load_eval_set()
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} eval items (hash {eval_set_hash()})")

    strategies = ["baseline", "structure"] if args.strategy == "all" else [args.strategy]
    summaries = [evaluate_strategy(items, s, run_ragas=not args.no_ragas, pause_s=args.pause) for s in strategies]

    print("\n" + "=" * 78 + "\ncomparison\n" + "=" * 78)
    print(build_comparison_table(summaries))

    if not args.no_mlflow:
        run_id = log_mlflow(summaries, experiment=args.experiment)
        print(f"\nMLflow: logged experiment {args.experiment!r} (parent run {run_id}). View with: mlflow ui")
    if args.langsmith_upload:
        push_langsmith_dataset(items)


if __name__ == "__main__":
    main()
