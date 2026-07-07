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
