"""Pure aggregation helpers for the eval run (Day 8-9) — no network, unit-testable.

Two jobs:
  1. Split runs into the RAGAS-scorable subset (the pipeline actually answered) vs
     the rest. RAGAS faithfulness/relevancy are undefined on a refusal, so refused
     and errored runs are excluded from RAGAS but still counted by the refusal
     metric (which scores the decision on the FULL set).
  2. Latency percentiles, split by branch — the README claims the refuse branch
     (~0.8s) is much faster than the answer branch (~3s); p95 per branch shows it.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def is_answered(r) -> bool:
    """True if the pipeline produced a real answer (not refused, not errored)."""
    return (not r.refused) and (not r.error) and bool(r.answer.strip())


def partition_for_ragas(results: Sequence) -> Tuple[list, list]:
    """(answered, not_answered): RAGAS scores only the answered subset."""
    answered = [r for r in results if is_answered(r)]
    other = [r for r in results if not is_answered(r)]
    return answered, other


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (p in [0,100]). Empty -> 0.0.

    Kept local (no numpy) — the eval set is tiny and this stays dependency-free.
    """
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"p must be in [0,100], got {p}")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    rank = (p / 100) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return float(xs[lo] + (xs[hi] - xs[lo]) * frac)


def latency_summary(results: Sequence) -> dict:
    """Mean + p95 latency overall and split by answer vs refuse branch."""
    all_lat = [r.latency_s for r in results]
    ans_lat = [r.latency_s for r in results if not r.refused and not r.error]
    ref_lat = [r.latency_s for r in results if r.refused]

    def _stats(xs: List[float]) -> dict:
        return {
            "n": len(xs),
            "mean": round(sum(xs) / len(xs), 4) if xs else 0.0,
            "p95": round(percentile(xs, 95), 4),
        }

    return {
        "overall": _stats(all_lat),
        "answer_branch": _stats(ans_lat),
        "refuse_branch": _stats(ref_lat),
    }


def _reference_unit(hit) -> str:
    metadata = getattr(hit, "metadata", {})
    unit_type = str(metadata.get("unit_type", "")).strip()
    number = str(metadata.get("number", "")).strip()
    if not unit_type or not number:
        return ""
    return f"{unit_type.title()} {number}".casefold()


def retrieval_quality(results: Sequence, candidate_k: int = 20, final_k: int = 5) -> dict:
    """Macro-average deterministic reference-unit metrics for answerable items."""
    rows = [
        result
        for result in results
        if not getattr(result, "error", "")
        and not result.item.should_refuse
        and result.item.reference_units
    ]
    if not rows:
        return {
            "n_scored": 0,
            "candidate_recall_at_20": 0.0,
            "hit_rate_at_5": 0.0,
            "reference_recall_at_5": 0.0,
            "mrr_at_5": 0.0,
            "ndcg_at_5": 0.0,
        }

    candidate_recalls = []
    hit_rates = []
    final_recalls = []
    reciprocal_ranks = []
    ndcgs = []
    for result in rows:
        references = {value.casefold() for value in result.item.reference_units}
        candidate_units = {
            _reference_unit(hit)
            for hit in result.candidate_hits[:candidate_k]
            if _reference_unit(hit)
        }
        candidate_recalls.append(len(references & candidate_units) / len(references))

        gains = []
        seen_units: set[str] = set()
        for hit in result.final_hits[:final_k]:
            unit = _reference_unit(hit)
            gain = 1 if unit in references and unit not in seen_units else 0
            gains.append(gain)
            if unit:
                seen_units.add(unit)
        matched = sum(gains)
        hit_rates.append(float(matched > 0))
        final_recalls.append(matched / len(references))
        first = next((rank for rank, gain in enumerate(gains, 1) if gain), None)
        reciprocal_ranks.append(1.0 / first if first else 0.0)
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
        ideal_count = min(len(references), final_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcgs.append(dcg / idcg if idcg else 0.0)

    mean = lambda values: round(sum(values) / len(values), 4)
    return {
        "n_scored": len(rows),
        "candidate_recall_at_20": mean(candidate_recalls),
        "hit_rate_at_5": mean(hit_rates),
        "reference_recall_at_5": mean(final_recalls),
        "mrr_at_5": mean(reciprocal_ranks),
        "ndcg_at_5": mean(ndcgs),
    }


def rerank_release_gate(off: dict, on: dict) -> dict:
    """Evaluate the paired structure off/on run against rollout thresholds.

    The caller must supply summaries produced from the same code version and eval
    set. Missing RAGAS metrics fail loudly instead of turning a ``--no-ragas``
    smoke run into a misleading rollout decision.
    """
    ragas_keys = {
        "llm_context_precision_with_reference",
        "context_recall",
        "faithfulness",
    }
    for label, summary in (("off", off), ("cohere", on)):
        missing = ragas_keys - set(summary.get("ragas", {}))
        if missing:
            raise ValueError(f"{label} summary lacks required RAGAS metrics: {sorted(missing)}")

    off_ragas, on_ragas = off["ragas"], on["ragas"]
    precision_delta = round((
        on_ragas["llm_context_precision_with_reference"]
        - off_ragas["llm_context_precision_with_reference"]
    ), 10)
    recall_delta = round(on_ragas["context_recall"] - off_ragas["context_recall"], 10)
    faithfulness_delta = round(on_ragas["faithfulness"] - off_ragas["faithfulness"], 10)
    latency_delta = round((
        on["latency"]["answer_branch"]["p95"]
        - off["latency"]["answer_branch"]["p95"]
    ), 10)
    healthy = (
        on.get("rerank", {}).get("fallback_count", 0) == 0
        and on.get("rerank", {}).get("applied_count", 0) > 0
    )

    def check(passed: bool, actual, requirement: str) -> dict:
        return {"passed": bool(passed), "actual": actual, "requirement": requirement}

    checks = {
        "context_precision_delta": check(
            precision_delta >= 0.03,
            round(precision_delta, 4),
            ">= +0.03",
        ),
        "context_recall_delta": check(
            recall_delta >= -0.02,
            round(recall_delta, 4),
            ">= -0.02",
        ),
        "faithfulness_delta": check(
            faithfulness_delta >= -0.01,
            round(faithfulness_delta, 4),
            ">= -0.01",
        ),
        "refusal_recall": check(
            on["refusal"]["recall"] == 1.0,
            on["refusal"]["recall"],
            "== 1.00",
        ),
        "under_refusals": check(
            on["refusal"]["false_negatives"] == 0,
            on["refusal"]["false_negatives"],
            "== 0",
        ),
        "healthy_reranker": check(
            healthy,
            {
                "fallback_count": on.get("rerank", {}).get("fallback_count", 0),
                "applied_count": on.get("rerank", {}).get("applied_count", 0),
            },
            "fallback_count == 0 and applied_count > 0",
        ),
        "answered_p95_latency_delta_s": check(
            latency_delta <= 1.0,
            round(latency_delta, 4),
            "<= +1.0s",
        ),
    }
    candidate_recall = on.get("retrieval", {}).get("candidate_recall_at_20", 0.0)
    limited = candidate_recall < 0.95
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "rerank_ceiling": {
            "limited": limited,
            "candidate_recall_at_20": candidate_recall,
            "next_step": (
                "prioritize hybrid retrieval before further reranker tuning"
                if limited
                else "rerank candidate ceiling is acceptable"
            ),
        },
    }
