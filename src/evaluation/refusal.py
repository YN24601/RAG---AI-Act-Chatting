"""Refusal-accuracy metric (Day 8-9) — pure functions, no network.

`refused` is the authoritative "no corpus basis" signal (README invariant); this
scores the pipeline's refusal *decision* separately from RAGAS answer quality.
The positive class is "should be refused", so:

  - false negative (should_refuse but answered) is the safety-critical case — the
    pipeline gave a legal answer with no grounding. Recall targets exactly this.
  - false positive (answerable but refused) is over-refusal — a usability cost.

Kept dependency-free (operates on plain bool lists) so it is unit-testable offline
and reusable by both the CLI and MLflow logging.
"""
from __future__ import annotations

from typing import List


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def refusal_scores(expected: List[bool], actual: List[bool]) -> dict:
    """Confusion matrix + accuracy/precision/recall/f1 for the refusal decision.

    Args:
        expected: `should_refuse` per item (True = the pipeline ought to refuse).
        actual:   `refused` per item (True = the pipeline did refuse).

    Positive class = refusal. Returns counts plus derived rates; `false_negatives`
    (should-refuse-but-answered) is the safety-critical figure to watch.
    """
    if len(expected) != len(actual):
        raise ValueError(f"length mismatch: expected {len(expected)} vs actual {len(actual)}")
    n = len(expected)
    tp = sum(1 for e, a in zip(expected, actual) if e and a)
    fp = sum(1 for e, a in zip(expected, actual) if not e and a)
    fn = sum(1 for e, a in zip(expected, actual) if e and not a)
    tn = sum(1 for e, a in zip(expected, actual) if not e and not a)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, n)

    return {
        "n": n,
        "true_positives": tp,   # correctly refused
        "false_positives": fp,  # over-refusal (answerable but refused)
        "false_negatives": fn,  # UNDER-refusal (should refuse but answered) — safety-critical
        "true_negatives": tn,   # correctly answered
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
