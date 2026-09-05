"""Network-free API contract tests for rerank provenance."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402

from api import app as app_mod  # noqa: E402
from api.schemas import AskRequest, QueryRequest  # noqa: E402
from generation.errors import PipelineError  # noqa: E402
from retrieval.reranker import RerankOutcome  # noqa: E402
from retrieval.retriever import Hit, RerankExecutionError  # noqa: E402


def _hit(rank: int, retrieval_rank: int, chunk_id: str, rerank_score=None) -> Hit:
    return Hit(
        rank=rank,
        retrieval_rank=retrieval_rank,
        score=0.8,
        rerank_score=rerank_score,
        chunk_id=chunk_id,
        text=f"text-{chunk_id}",
        metadata={"context_header": f"Article {retrieval_rank}"},
    )


def test_ask_serializes_rerank_metadata_and_exact_answer_hits(monkeypatch):
    first = _hit(1, 2, "a", 0.95)
    second = _hit(2, 1, "b", 0.72)
    monkeypatch.setattr(
        app_mod,
        "answer_question",
        lambda *args, **kwargs: {
            "answer": "answer",
            "refused": False,
            "grade": "relevant",
            "hits": [first, second],
            "answer_hits": [second],
            "used_hits": 1,
            "rerank_status": "applied",
            "rerank_model": "rerank-test",
            "rerank_latency_ms": 13.2,
        },
    )

    response = app_mod.ask(AskRequest(question="q"))

    assert response.rerank_status == "applied"
    assert response.rerank_model == "rerank-test"
    assert response.sources[0].retrieval_rank == 2
    assert response.sources[0].rerank_score == 0.95
    assert response.sources[0].used is False
    assert response.sources[1].used is True


def test_query_accepts_rerank_override_and_returns_outcome(monkeypatch):
    hit = _hit(1, 3, "c", 0.88)

    class FakeRetriever:
        def search_with_outcome(self, question, **kwargs):
            assert kwargs["rerank"] is True
            return RerankOutcome([hit], "applied", "rerank-test", 9.5)

    monkeypatch.setattr(app_mod, "_get_retriever", lambda strategy: FakeRetriever())
    response = app_mod.query(QueryRequest(question="q", rerank=True))

    assert response.rerank_status == "applied"
    assert response.rerank_latency_ms == 9.5
    assert response.hits[0].retrieval_rank == 3
    assert response.hits[0].rerank_score == 0.88


def test_query_maps_non_transient_reranker_error_to_rerank_503_stage(monkeypatch):
    class BrokenRetriever:
        def search_with_outcome(self, *args, **kwargs):
            raise RerankExecutionError("invalid reranker configuration")

    monkeypatch.setattr(app_mod, "_get_retriever", lambda strategy: BrokenRetriever())

    with pytest.raises(PipelineError) as exc_info:
        app_mod.query(QueryRequest(question="q", rerank=True))

    assert exc_info.value.stage == "rerank"
