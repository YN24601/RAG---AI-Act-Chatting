"""RAGAS, refusal, retrieval, and latency evaluation across rerank modes.

Runs the production RAG pipeline over the committed eval set across the selected
chunking and reranking axes, scores the answered subset with RAGAS, scores the
refusal decision on the full set, logs everything to MLflow as a parent compare
run with nested per-configuration runs, and optionally enforces the rollout gate.

Usage:
    python scripts/evaluate.py --strategy all
    python scripts/evaluate.py --strategy structure --rerank all --enforce-rerank-gate
    python scripts/evaluate.py --strategy structure --limit 3        # smoke test
    python scripts/evaluate.py --strategy all --langsmith-upload
    python scripts/evaluate.py --strategy structure --no-ragas       # refusal+latency only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.aggregate import (  # noqa: E402
    latency_summary,
    partition_for_ragas,
    rerank_release_gate,
    retrieval_quality,
)
from evaluation.harness import run_over_set  # noqa: E402
from evaluation.refusal import refusal_scores  # noqa: E402
from evaluation.schema import eval_set_hash, load_eval_set  # noqa: E402
from evaluation.tracking import build_comparison_table, log_mlflow, push_langsmith_dataset  # noqa: E402
from generation import config as gconfig  # noqa: E402
from retrieval import config as rconfig  # noqa: E402


TRIAL_RERANK_PAUSE_S = 6.5


def evaluation_configs(strategy: str, rerank: str) -> list[tuple[str, bool]]:
    strategies = ["baseline", "structure"] if strategy == "all" else [strategy]
    rerank_modes = [False, True] if rerank == "all" else [rerank == "cohere"]
    return [(selected_strategy, enabled) for selected_strategy in strategies for enabled in rerank_modes]


def effective_pause(rerank: bool, requested_pause_s: float) -> float:
    """Respect Cohere trial-key pacing for reproducible evaluation runs."""
    return max(requested_pause_s, TRIAL_RERANK_PAUSE_S) if rerank else requested_pause_s


def paired_structure_gate(summaries: list[dict]) -> dict | None:
    """Build a release decision when both paired structure runs are present."""
    structure = {s["params"]["rerank"]: s for s in summaries if s["strategy"] == "structure"}
    if not {"off", "cohere"} <= structure.keys():
        return None
    return rerank_release_gate(structure["off"], structure["cohere"])


def print_release_gate(gate: dict) -> None:
    verdict = "PASS" if gate["passed"] else "FAIL"
    print(f"\nrerank release gate: {verdict}")
    for name, result in gate["checks"].items():
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"  {mark:<4} {name}: {result['actual']} ({result['requirement']})")
    ceiling = gate["rerank_ceiling"]
    print(
        f"  candidate recall@20: {ceiling['candidate_recall_at_20']:.3f} — "
        f"{ceiling['next_step']}"
    )


def evaluate_strategy(
    items,
    strategy: str,
    run_ragas: bool,
    rerank: bool = False,
    pause_s: float = 0.0,
) -> dict:
    rerank_name = "cohere" if rerank else "off"
    label = f"{strategy}+rerank-{rerank_name}"
    print(f"\n=== configuration: {label} ===")
    results = run_over_set(
        items,
        strategy,
        pause_s=effective_pause(rerank, pause_s),
        rerank=rerank,
    )

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
    retrieval = retrieval_quality(results) if strategy == "structure" else {}
    rerank_latency = [r.rerank_latency_ms for r in results if r.rerank_status == "applied"]
    rerank_stats = {
        "applied_count": sum(r.rerank_status == "applied" for r in results),
        "fallback_count": sum(r.rerank_status == "fallback" for r in results),
        "latency_ms_mean": round(sum(rerank_latency) / len(rerank_latency), 2) if rerank_latency else 0.0,
    }

    params = {
        "strategy": strategy,
        "k": rconfig.DEFAULT_K,
        "top_n": gconfig.ANSWER_TOP_N,
        "rerank": rerank_name,
        "rerank_model": rconfig.RERANK_MODEL if rerank else "none",
        "rerank_timeout_s": rconfig.RERANK_TIMEOUT_S,
        "embed_model": rconfig.EMBED_MODEL,
        "gen_model": gconfig.GEN_MODEL,
        "grade_min_score": gconfig.GRADE_MIN_SCORE,
        "ragas_version": rv,
        "eval_set_hash": eval_set_hash(),
        "n_items": len(results),
    }
    return {"strategy": strategy, "label": label, "params": params, "ragas": ragas,
            "refusal": refusal, "latency": latency, "retrieval": retrieval,
            "rerank": rerank_stats, "results": results}


def main() -> None:
    ap = argparse.ArgumentParser(description="EU AI Act RAG evaluation (Day 8-9)")
    ap.add_argument("--strategy", choices=["baseline", "structure", "all"], default="all")
    ap.add_argument("--rerank", choices=["off", "cohere", "all"], default="off")
    ap.add_argument("--limit", type=int, default=None, help="only the first N items (smoke test)")
    ap.add_argument("--no-ragas", action="store_true", help="skip RAGAS (refusal + latency only)")
    ap.add_argument("--no-mlflow", action="store_true", help="don't log to MLflow")
    ap.add_argument(
        "--enforce-rerank-gate",
        action="store_true",
        help="exit non-zero unless the paired structure off/cohere run passes rollout thresholds",
    )
    ap.add_argument("--langsmith-upload", action="store_true", help="persist eval set as LangSmith dataset")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between items (smooths Mistral rate limit)")
    ap.add_argument("--experiment", default="aiact-rag-eval")
    args = ap.parse_args()
    if args.enforce_rerank_gate and (
        args.rerank != "all" or args.strategy not in {"structure", "all"} or args.no_ragas
    ):
        ap.error("--enforce-rerank-gate requires structure, --rerank all, and RAGAS enabled")

    items = load_eval_set()
    if args.limit:
        items = items[: args.limit]
    print(f"loaded {len(items)} eval items (hash {eval_set_hash()})")

    configs = evaluation_configs(args.strategy, args.rerank)
    summaries = [
        evaluate_strategy(
            items,
            strategy,
            run_ragas=not args.no_ragas,
            rerank=rerank,
            pause_s=args.pause,
        )
        for strategy, rerank in configs
    ]

    print("\n" + "=" * 78 + "\ncomparison\n" + "=" * 78)
    print(build_comparison_table(summaries))

    gate = None
    if args.rerank == "all" and not args.no_ragas:
        gate = paired_structure_gate(summaries)
        if gate:
            print_release_gate(gate)

    if not args.no_mlflow:
        run_id = log_mlflow(summaries, experiment=args.experiment)
        print(f"\nMLflow: logged experiment {args.experiment!r} (parent run {run_id}). View with: mlflow ui")
    if args.langsmith_upload:
        push_langsmith_dataset(items)
    if args.enforce_rerank_gate and (not gate or not gate["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
