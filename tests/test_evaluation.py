"""Network-free unit tests for the evaluation layer (Day 8-9).

No Mistral / Qdrant / RAGAS / MLflow calls: only pure helpers, the eval-set
schema/loader, the run-partition logic (with an injected fake pipeline runner),
and the comparison-table builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from evaluation.aggregate import (  # noqa: E402
    is_answered,
    latency_summary,
    partition_for_ragas,
    percentile,
    rerank_release_gate,
    retrieval_quality,
)
from evaluation.harness import run_over_set  # noqa: E402
from evaluation.refusal import refusal_scores  # noqa: E402
from evaluation.schema import EVAL_SET_PATH, EvalItem, eval_set_hash, load_eval_set  # noqa: E402
from evaluation.tracking import build_comparison_table  # noqa: E402
from scripts.evaluate import effective_pause, evaluation_configs, paired_structure_gate  # noqa: E402


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
        def __init__(self, t, unit_type="article", number="5"):
            self.text = t
            self.metadata = {"unit_type": unit_type, "number": number}
            self.chunk_id = t
    return {"answer": answer, "refused": refused, "grade": "relevant" if not refused else "irrelevant",
            "candidates": [_H("candidate")],
            "hits": [_H("not-sent-to-generation"), *[_H(t) for t in hits_text]],
            "answer_hits": [_H(t) for t in hits_text],
            "rerank_status": "applied", "rerank_model": "rerank-test",
            "rerank_latency_ms": 7.5}


def test_run_over_set_and_partition_offline():
    items = [
        EvalItem(id="ans", question="q1", category="prohibited", ground_truth="g", should_refuse=False),
        EvalItem(id="ref", question="q2", category="out_of_scope", ground_truth="g", should_refuse=True),
    ]

    def runner(question, strategy, rerank=None):
        if question == "q1":
            return _fake_state("Article 5 says ...", False, ["ctx a", "ctx b"])
        return _fake_state("<refusal>", True, [])

    results = run_over_set(items, "structure", runner=runner, progress=False)
    assert [r.refused for r in results] == [False, True]
    assert results[0].contexts == ["ctx a", "ctx b"]
    assert [h.text for h in results[0].candidate_hits] == ["candidate"]
    assert results[0].rerank_status == "applied"

    answered, other = partition_for_ragas(results)
    assert [r.item.id for r in answered] == ["ans"]
    assert [r.item.id for r in other] == ["ref"]
    assert is_answered(results[0]) and not is_answered(results[1])


def test_run_over_set_captures_errors_without_aborting():
    items = [EvalItem(id="boom", question="q", category="prohibited", ground_truth="g")]

    def runner(question, strategy, rerank=None):
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

    def runner(question, strategy, rerank=None):
        return _fake_state("ans", question == "q2", ["c"])

    results = run_over_set(items, "structure", runner=runner, progress=False)
    summ = latency_summary(results)
    assert summ["answer_branch"]["n"] == 1
    assert summ["refuse_branch"]["n"] == 1


def test_retrieval_quality_scores_reference_units_at_candidate_and_final_depths():
    class H:
        def __init__(self, unit_type, number):
            self.metadata = {"unit_type": unit_type, "number": number}

    item_a = EvalItem(
        id="a", question="q", category="high_risk", ground_truth="g",
        reference_units=["Article 6", "Annex III"],
    )
    item_b = EvalItem(
        id="b", question="q", category="definition", ground_truth="g",
        reference_units=["Article 3"],
    )

    class Result:
        pass

    a = Result()
    a.item, a.error = item_a, ""
    a.candidate_hits = [H("article", "6"), H("annex", "III")]
    a.final_hits = [H("article", "6"), H("article", "99"), H("annex", "III")]
    b = Result()
    b.item, b.error = item_b, ""
    b.candidate_hits = [H("article", "3")]
    b.final_hits = [H("article", "99")]

    metrics = retrieval_quality([a, b])

    assert metrics["n_scored"] == 2
    assert metrics["candidate_recall_at_20"] == 1.0
    assert metrics["hit_rate_at_5"] == 0.5
    assert metrics["reference_recall_at_5"] == 0.5
    assert metrics["mrr_at_5"] == 0.5
    assert 0.45 < metrics["ndcg_at_5"] < 0.5


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


def test_evaluation_matrix_expands_strategy_and_rerank_axes():
    assert evaluation_configs("all", "all") == [
        ("baseline", False),
        ("baseline", True),
        ("structure", False),
        ("structure", True),
    ]
    assert evaluation_configs("structure", "cohere") == [("structure", True)]


def test_cohere_evaluation_enforces_trial_safe_pacing():
    assert effective_pause(False, 0.0) == 0.0
    assert effective_pause(True, 0.0) == 6.5
    assert effective_pause(True, 8.0) == 8.0


def _release_summary(*, rerank, precision, recall, faithfulness, refusal_recall=1.0,
                     false_negatives=0, answer_p95=3.0, candidate_recall=0.97,
                     fallbacks=0, applied=37):
    return {
        "params": {"rerank": rerank},
        "ragas": {
            "llm_context_precision_with_reference": precision,
            "context_recall": recall,
            "faithfulness": faithfulness,
        },
        "refusal": {"recall": refusal_recall, "false_negatives": false_negatives},
        "latency": {"answer_branch": {"p95": answer_p95}},
        "retrieval": {"candidate_recall_at_20": candidate_recall},
        "rerank": {"fallback_count": fallbacks, "applied_count": applied},
    }


def test_rerank_release_gate_passes_all_acceptance_thresholds():
    off = _release_summary(rerank="off", precision=0.85, recall=0.82,
                           faithfulness=0.98, answer_p95=3.2)
    on = _release_summary(rerank="cohere", precision=0.89, recall=0.81,
                          faithfulness=0.975, answer_p95=4.0)

    gate = rerank_release_gate(off, on)

    assert gate["passed"] is True
    assert all(check["passed"] for check in gate["checks"].values())
    assert gate["rerank_ceiling"]["limited"] is False


def test_rerank_release_gate_fails_and_records_low_candidate_recall_ceiling():
    off = _release_summary(rerank="off", precision=0.85, recall=0.82,
                           faithfulness=0.98, answer_p95=3.2)
    on = _release_summary(
        rerank="cohere", precision=0.87, recall=0.79, faithfulness=0.95,
        refusal_recall=0.9, false_negatives=1, answer_p95=4.5,
        candidate_recall=0.9, fallbacks=1,
    )

    gate = rerank_release_gate(off, on)

    assert gate["passed"] is False
    assert gate["checks"]["context_precision_delta"]["passed"] is False
    assert gate["checks"]["healthy_reranker"]["passed"] is False
    assert gate["rerank_ceiling"] == {
        "limited": True,
        "candidate_recall_at_20": 0.9,
        "next_step": "prioritize hybrid retrieval before further reranker tuning",
    }


def test_paired_structure_gate_ignores_baseline_and_requires_both_modes():
    off = _release_summary(rerank="off", precision=0.85, recall=0.82,
                           faithfulness=0.98, answer_p95=3.2)
    on = _release_summary(rerank="cohere", precision=0.89, recall=0.81,
                          faithfulness=0.975, answer_p95=4.0)
    off["strategy"] = on["strategy"] = "structure"
    baseline = dict(off, strategy="baseline")

    assert paired_structure_gate([baseline, off]) is None
    assert paired_structure_gate([baseline, off, on])["passed"] is True
