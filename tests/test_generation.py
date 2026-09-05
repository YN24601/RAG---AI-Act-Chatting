"""Network-free unit tests for the generation/orchestration layer (Day 5).

No Mistral / Qdrant / LangSmith calls: only pure helpers and graph assembly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from generation import config  # noqa: E402
from generation import graph as graph_mod  # noqa: E402
from generation.errors import PipelineError  # noqa: E402
from generation.graph import build_graph, finalize_answer  # noqa: E402
from generation.grade import score_gate, select_answer_hits  # noqa: E402
from generation.prompts import format_context  # noqa: E402
from retrieval import config as retrieval_config  # noqa: E402
from retrieval.reranker import RerankOutcome  # noqa: E402
from retrieval.retriever import Hit  # noqa: E402


def _hit(rank, score, header="Article 5 — Prohibited AI practices", chapter="Chapter II"):
    return Hit(
        rank=rank,
        score=score,
        chunk_id=f"article-5-{rank}",
        text=f"Some provision text {rank}.",
        metadata={"context_header": header, "chapter": chapter, "unit_type": "article"},
    )


def test_score_gate_empty_is_false():
    assert score_gate([]) is False


def test_score_gate_below_threshold_is_false():
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE - 0.05)]) is False


def test_score_gate_at_or_above_threshold_is_true():
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE)]) is True
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE + 0.2)]) is True


def test_select_answer_hits_empty():
    assert select_answer_hits([]) == []


def test_select_answer_hits_drops_weak_tail():
    # Top 0.85; a 0.60 tail hit is >REL_DROP below it and below the relative floor
    # (0.75) -> dropped. The 0.80 hit is within the band -> kept.
    hits = [_hit(1, 0.85), _hit(2, 0.80), _hit(3, 0.60)]
    kept = select_answer_hits(hits)
    assert [h.rank for h in kept] == [1, 2]


def test_select_answer_hits_absolute_floor():
    # Top is itself low (0.60) so the absolute floor (0.55), not the relative band,
    # governs: the 0.50 hit is below it and dropped.
    hits = [_hit(1, 0.60), _hit(2, 0.50)]
    kept = select_answer_hits(hits)
    assert [h.rank for h in kept] == [1]


def test_select_answer_hits_always_keeps_top():
    # Even a single weak hit (below the absolute floor) is preserved — the score
    # gate decides whether we generate at all; selection never returns empty.
    assert [h.rank for h in select_answer_hits([_hit(1, 0.10)])] == [1]


def test_select_answer_hits_keeps_all_when_tight():
    hits = [_hit(1, 0.90), _hit(2, 0.88), _hit(3, 0.85)]
    assert len(select_answer_hits(hits)) == 3


def test_score_gates_raise_when_distance_not_calibrated(monkeypatch):
    # score_gate / select_answer_hits all assume higher-is-better Cosine scores; a
    # DISTANCE change must surface as an error here, not silently invert the gates.
    monkeypatch.setattr(retrieval_config, "DISTANCE", "Euclid")
    with pytest.raises(RuntimeError, match="calibrated for DISTANCE"):
        score_gate([_hit(1, 0.90)])
    with pytest.raises(RuntimeError, match="calibrated for DISTANCE"):
        select_answer_hits([_hit(1, 0.90), _hit(2, 0.80)])


def test_format_context_empty():
    assert format_context([]) == "(no context retrieved)"


def test_format_context_carries_headers_in_rank_order():
    ctx = format_context([_hit(1, 0.9, header="Article 5 — Prohibited AI practices"),
                          _hit(2, 0.8, header="Article 6 — Classification rules")])
    assert "Article 5" in ctx and "Article 6" in ctx
    assert "Chapter II" in ctx
    assert ctx.index("[1]") < ctx.index("[2]")  # preserves rank order


def test_refusal_text_is_stable_and_nonempty():
    assert config.REFUSAL_TEXT
    assert "cannot confirm" in config.REFUSAL_TEXT.lower()


def test_finalize_answer_passes_real_answer_through():
    ans, refused = finalize_answer("Under Article 5, the following practices are prohibited...")
    assert refused is False
    assert ans.startswith("Under Article 5")


def test_finalize_answer_maps_sentinel_to_verbatim_refusal():
    # In-generation refusal must be verbatim REFUSAL_TEXT and flagged refused=True,
    # not an LLM paraphrase mislabeled as a successful answer (the bug this fixes).
    for raw in (config.INSUFFICIENT_SENTINEL, f"  {config.INSUFFICIENT_SENTINEL}\n", "insufficient_context"):
        ans, refused = finalize_answer(raw)
        assert refused is True
        assert ans == config.REFUSAL_TEXT


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"retrieve", "score_gate", "rerank", "grade", "generate", "refuse"} <= nodes


def test_retrieve_wraps_network_error_as_pipeline_error(monkeypatch):
    # A raw Qdrant failure must surface as a controlled PipelineError (stage tagged),
    # not leak the client's stack out of the graph.
    def boom(*a, **k):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(graph_mod, "_run_recall", boom)
    with pytest.raises(PipelineError) as ei:
        graph_mod.retrieve({"question": "q", "strategy": "structure"})
    assert ei.value.stage == "retrieve"
    assert isinstance(ei.value.__cause__, ConnectionError)  # original chained, not lost


def test_grade_falls_back_to_score_gate_when_llm_unavailable(monkeypatch):
    # The LLM grader is a soft refinement on top of the score gate; if Mistral is
    # down the request degrades to relevant (score gate already passed), not crash.
    def boom(*a, **k):
        raise TimeoutError("mistral timeout")

    monkeypatch.setattr(graph_mod, "llm_grade", boom)
    out = graph_mod.grade({"question": "q", "hits": [_hit(1, 0.90)]})
    assert out["grade"] == "relevant"
    assert "fell back to score gate" in out["grade_reason"]


def test_score_gate_refuses_before_reranker_is_called(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("reranker must not run for weak vector recall")

    monkeypatch.setattr(
        graph_mod,
        "_run_recall",
        lambda *args, **kwargs: [_hit(1, config.GRADE_MIN_SCORE - 0.1)],
    )
    monkeypatch.setattr(graph_mod, "_run_rerank", must_not_run)

    out = graph_mod.answer_question("q", rerank=True)

    assert out["score_gate_passed"] is False
    assert out["grade"] == "irrelevant"
    assert out["rerank_status"] == "off"
    assert out["refused"] is True


def test_rerank_node_surfaces_outcome_metadata(monkeypatch):
    hits = [_hit(1, 0.9), _hit(2, 0.8)]
    reranked = [_hit(1, 0.8), _hit(2, 0.9)]
    reranked[0].retrieval_rank = 2
    reranked[0].rerank_score = 0.98
    outcome = RerankOutcome(reranked, "applied", "rerank-test", 12.5)
    monkeypatch.setattr(graph_mod, "_run_rerank", lambda *args, **kwargs: outcome)

    out = graph_mod.rerank({
        "question": "q",
        "strategy": "structure",
        "candidates": hits,
        "rerank": True,
    })

    assert out["hits"] == reranked
    assert out["rerank_status"] == "applied"
    assert out["rerank_model"] == "rerank-test"
    assert out["rerank_latency_ms"] == 12.5


def test_non_transient_rerank_error_becomes_pipeline_error(monkeypatch):
    monkeypatch.setattr(
        graph_mod,
        "_run_rerank",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("invalid key")),
    )
    with pytest.raises(PipelineError) as ei:
        graph_mod.rerank({
            "question": "q",
            "strategy": "structure",
            "candidates": [_hit(1, 0.9)],
            "rerank": True,
        })
    assert ei.value.stage == "rerank"


def test_reranked_hits_all_ground_answer_but_vector_fallback_keeps_score_trim():
    hits = [_hit(1, 0.90), _hit(2, 0.70), _hit(3, 0.40)]
    assert graph_mod.select_grounding_hits(hits, "applied") == hits
    assert [h.rank for h in graph_mod.select_grounding_hits(hits, "fallback")] == [1]


@pytest.mark.parametrize(
    ("status", "expected_ranks"),
    [("applied", [1, 2, 3]), ("fallback", [1])],
)
def test_generate_prompt_and_answer_hits_use_exact_selected_contexts(
    monkeypatch, status, expected_ranks
):
    hits = [_hit(1, 0.90), _hit(2, 0.70), _hit(3, 0.40)]
    captured = {}

    class FakeChain:
        def __or__(self, other):
            return self

        def invoke(self, payload):
            captured.update(payload)
            return "grounded answer"

    monkeypatch.setattr(graph_mod, "ANSWER_PROMPT", FakeChain())
    monkeypatch.setattr(graph_mod, "get_chat_llm", lambda: object())

    out = graph_mod.generate({
        "question": "q",
        "hits": hits,
        "rerank_status": status,
    })

    assert [hit.rank for hit in out["answer_hits"]] == expected_ranks
    for rank in expected_ranks:
        assert f"Some provision text {rank}." in captured["context"]
    for rank in {1, 2, 3} - set(expected_ranks):
        assert f"Some provision text {rank}." not in captured["context"]


def test_generate_wraps_network_error_as_pipeline_error(monkeypatch):
    # A raw Mistral failure during answer generation -> controlled PipelineError,
    # never a fabricated answer and never mislabeled as a refusal.
    def boom(*a, **k):
        raise ConnectionError("mistral down")

    monkeypatch.setattr(graph_mod, "get_chat_llm", boom)
    with pytest.raises(PipelineError) as ei:
        graph_mod.generate({"question": "q", "hits": [_hit(1, 0.90)]})
    assert ei.value.stage == "generate"
