"""Network-free unit tests for the evaluation layer (Day 8-9).

No Mistral / Qdrant / RAGAS / MLflow calls: only pure helpers, the eval-set
schema/loader, the run-partition logic (with an injected fake pipeline runner),
and the comparison-table builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from evaluation.aggregate import is_answered, latency_summary, partition_for_ragas, percentile  # noqa: E402
from evaluation.harness import run_over_set  # noqa: E402
from evaluation.refusal import refusal_scores  # noqa: E402
from evaluation.schema import EVAL_SET_PATH, EvalItem, eval_set_hash, load_eval_set  # noqa: E402
from evaluation.tracking import build_comparison_table  # noqa: E402


# ---------------- refusal metric ----------------

def test_refusal_all_correct():
    s = refusal_scores(expected=[True, False, True, False], actual=[True, False, True, False])
    assert s["accuracy"] == 1.0 and s["precision"] == 1.0 and s["recall"] == 1.0 and s["f1"] == 1.0
    assert s["false_negatives"] == 0 and s["false_positives"] == 0


def test_refusal_under_refusal_is_false_negative():
    # should refuse but answered -> the safety-critical FN
    s = refusal_scores(expected=[True, True], actual=[False, True])
    assert s["false_negatives"] == 1
    assert s["recall"] == 0.5


def test_refusal_over_refusal_is_false_positive():
    # answerable but refused -> FP (usability cost)
    s = refusal_scores(expected=[False, False], actual=[True, False])
    assert s["false_positives"] == 1
    assert s["precision"] == 0.0  # no true positives


def test_refusal_all_wrong():
    s = refusal_scores(expected=[True, False], actual=[False, True])
    assert s["accuracy"] == 0.0 and s["f1"] == 0.0


def test_refusal_length_mismatch_raises():
    with pytest.raises(ValueError):
        refusal_scores(expected=[True], actual=[True, False])


# ---------------- percentile / latency ----------------

def test_percentile_empty_and_single():
    assert percentile([], 95) == 0.0
    assert percentile([2.5], 95) == 2.5


def test_percentile_interpolates():
    assert percentile([0, 10], 50) == 5.0
    assert percentile([1, 2, 3, 4], 100) == 4.0
    assert percentile([1, 2, 3, 4], 0) == 1.0


def test_percentile_out_of_range_raises():
    with pytest.raises(ValueError):
        percentile([1, 2], 150)


# ---------------- eval-set schema + loader ----------------

def test_evalitem_rejects_bad_category():
    with pytest.raises(Exception):
        EvalItem(id="x", question="q", category="nonsense", ground_truth="g")


def test_evalitem_defaults():
    it = EvalItem(id="x", question="q", category="prohibited", ground_truth="g")
    assert it.should_refuse is False and it.reference_units == []


def test_load_eval_set_roundtrip(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = [
        '{"id":"a","question":"q1","category":"prohibited","ground_truth":"g1","should_refuse":false}',
        "",  # blank lines are skipped
        '{"id":"b","question":"q2","category":"out_of_scope","ground_truth":"g2","should_refuse":true}',
    ]
    p.write_text("\n".join(rows), encoding="utf-8")
    items = load_eval_set(p)
    assert [i.id for i in items] == ["a", "b"]
    assert items[1].should_refuse is True


def test_load_eval_set_duplicate_id_raises(tmp_path):
    p = tmp_path / "dup.jsonl"
    p.write_text(
        '{"id":"a","question":"q","category":"prohibited","ground_truth":"g"}\n'
        '{"id":"a","question":"q2","category":"prohibited","ground_truth":"g2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        load_eval_set(p)


def test_load_eval_set_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eval_set(tmp_path / "nope.jsonl")


def test_committed_eval_set_is_valid():
    """The real committed eval set loads, has a healthy size, and includes traps."""
    items = load_eval_set()
    assert len(items) >= 30
    assert any(i.should_refuse for i in items), "need refusal traps"
    assert any(not i.should_refuse for i in items), "need answerable items"
    assert len(eval_set_hash()) == 16


# ---------------- run partition (offline, injected runner) ----------------

def _fake_state(answer, refused, hits_text):
    class _H:
        def __init__(self, t):
            self.text = t
    return {"answer": answer, "refused": refused, "grade": "relevant" if not refused else "irrelevant",
            "hits": [_H(t) for t in hits_text]}


def test_run_over_set_and_partition_offline():
    items = [
        EvalItem(id="ans", question="q1", category="prohibited", ground_truth="g", should_refuse=False),
        EvalItem(id="ref", question="q2", category="out_of_scope", ground_truth="g", should_refuse=True),
    ]

    def runner(question, strategy):
        if question == "q1":
            return _fake_state("Article 5 says ...", False, ["ctx a", "ctx b"])
        return _fake_state("<refusal>", True, [])

    results = run_over_set(items, "structure", runner=runner, progress=False)
    assert [r.refused for r in results] == [False, True]
    assert results[0].contexts == ["ctx a", "ctx b"]

    answered, other = partition_for_ragas(results)
    assert [r.item.id for r in answered] == ["ans"]
    assert [r.item.id for r in other] == ["ref"]
    assert is_answered(results[0]) and not is_answered(results[1])


def test_run_over_set_captures_errors_without_aborting():
    items = [EvalItem(id="boom", question="q", category="prohibited", ground_truth="g")]

    def runner(question, strategy):
        raise RuntimeError("qdrant down")

    results = run_over_set(items, "structure", runner=runner, progress=False)
    assert len(results) == 1
    assert results[0].error.startswith("RuntimeError")
    assert not is_answered(results[0])  # errored -> excluded from RAGAS


def test_latency_summary_splits_by_branch():
    items = [
        EvalItem(id="a", question="q1", category="prohibited", ground_truth="g", should_refuse=False),
        EvalItem(id="b", question="q2", category="out_of_scope", ground_truth="g", should_refuse=True),
    ]

    def runner(question, strategy):
        return _fake_state("ans", question == "q2", ["c"])

    results = run_over_set(items, "structure", runner=runner, progress=False)
    summ = latency_summary(results)
    assert summ["answer_branch"]["n"] == 1
    assert summ["refuse_branch"]["n"] == 1


# ---------------- comparison table ----------------

def test_build_comparison_table_shape():
    summaries = [{
        "strategy": s,
        "ragas": {"faithfulness": 0.9, "answer_relevancy": 0.8,
                  "llm_context_precision_with_reference": p, "context_recall": 0.7},
        "refusal": {"accuracy": 0.9, "recall": 1.0, "false_negatives": 0},
        "latency": {"answer_branch": {"p95": 3.1}, "refuse_branch": {"p95": 0.8}},
    } for s, p in [("baseline", 0.6), ("structure", 0.8)]]
    table = build_comparison_table(summaries)
    assert "| metric | baseline | structure |" in table
    assert "context_precision" in table
    assert "0.600" in table and "0.800" in table  # per-strategy precision cells
