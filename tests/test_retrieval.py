"""Network-free unit tests for the retrieval layer (no Qdrant/Mistral calls)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest  # noqa: E402
import httpx  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from retrieval import config  # noqa: E402
from retrieval.index import to_documents  # noqa: E402
from retrieval.reranker import (  # noqa: E402
    CohereReranker,
    RerankProtocolError,
    is_transient_rerank_error,
)
from retrieval.retriever import Hit, RerankExecutionError, Retriever, build_filter  # noqa: E402

_CHUNKS = [
    {"chunk_id": "article-6", "text": "Article 6 — …", "strategy": "structure",
     "metadata": {"unit_type": "article", "number": "6", "number_int": 6}},
    {"chunk_id": "baseline-0001", "text": "some text", "strategy": "baseline",
     "metadata": {"chunk_index": 1}},
]


def test_reranker_boundary_module_exists():
    assert importlib.util.find_spec("retrieval.reranker") is not None


def _hit(rank: int, score: float, chunk_id: str) -> Hit:
    return Hit(
        rank=rank,
        retrieval_rank=rank,
        score=score,
        rerank_score=None,
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        metadata={"context_header": f"Article {rank}", "chapter": "Chapter I"},
    )


class _FakeRerankClient:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def rerank(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(results=self.results)


def test_cohere_reranker_maps_scores_and_preserves_vector_provenance():
    candidates = [_hit(1, 0.91, "a"), _hit(2, 0.82, "b"), _hit(3, 0.77, "c")]
    client = _FakeRerankClient([
        SimpleNamespace(index=2, relevance_score=0.98),
        SimpleNamespace(index=0, relevance_score=0.71),
    ])
    reranker = CohereReranker(client=client, model="rerank-test")

    outcome = reranker.rerank("Which provision applies?", candidates, top_n=2)

    assert outcome.status == "applied"
    assert outcome.model == "rerank-test"
    assert [h.chunk_id for h in outcome.hits] == ["c", "a"]
    assert [h.rank for h in outcome.hits] == [1, 2]
    assert [h.retrieval_rank for h in outcome.hits] == [3, 1]
    assert [h.score for h in outcome.hits] == [0.77, 0.91]
    assert [h.rerank_score for h in outcome.hits] == [0.98, 0.71]
    assert client.calls[0]["documents"][0].startswith("Provision: Article 1")
    assert client.calls[0]["top_n"] == 2


def test_cohere_reranker_rejects_invalid_top_n_without_calling_provider():
    client = _FakeRerankClient([])
    reranker = CohereReranker(client=client, model="rerank-test")
    with pytest.raises(ValueError, match="top_n"):
        reranker.rerank("q", [_hit(1, 0.9, "a")], top_n=0)
    assert client.calls == []


@pytest.mark.parametrize(
    "results",
    [
        [],
        [SimpleNamespace(index=3, relevance_score=0.8)],
        [
            SimpleNamespace(index=0, relevance_score=0.8),
            SimpleNamespace(index=0, relevance_score=0.7),
        ],
    ],
)
def test_cohere_reranker_rejects_invalid_provider_indices(results):
    reranker = CohereReranker(client=_FakeRerankClient(results), model="rerank-test")
    with pytest.raises(RerankProtocolError):
        reranker.rerank("q", [_hit(1, 0.9, "a")], top_n=1)


@pytest.mark.parametrize("status", [429, 500, 503, 504])
def test_transient_rerank_status_codes_are_classified_for_fallback(status):
    error = RuntimeError("provider unavailable")
    error.status_code = status
    assert is_transient_rerank_error(error) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 498])
def test_configuration_rerank_status_codes_are_not_transient(status):
    error = RuntimeError("bad configuration")
    error.status_code = status
    assert is_transient_rerank_error(error) is False


@pytest.mark.parametrize("error", [httpx.TimeoutException("timeout"), httpx.ConnectError("down")])
def test_httpx_transport_errors_are_transient(error):
    assert is_transient_rerank_error(error) is True


class _FakeVectorStore:
    def __init__(self, scored):
        self.scored = scored
        self.calls = []

    def similarity_search_with_score(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.scored


def _bare_retriever(scored=(), reranker=None) -> Retriever:
    retriever = object.__new__(Retriever)
    retriever.k = 20
    retriever.vs = _FakeVectorStore(list(scored))
    retriever._reranker_client = reranker
    return retriever


def test_recall_populates_vector_rank_and_score_without_reranking():
    scored = [
        (Document(page_content="A", metadata={"chunk_id": "a"}), 0.91),
        (Document(page_content="B", metadata={"chunk_id": "b"}), 0.82),
    ]
    retriever = _bare_retriever(scored)

    hits = retriever.recall("question", k=20)

    assert [(h.rank, h.retrieval_rank, h.score, h.rerank_score) for h in hits] == [
        (1, 1, 0.91, None),
        (2, 2, 0.82, None),
    ]


def test_rerank_off_returns_vector_prefix_without_provider_call():
    class MustNotRun:
        def rerank(self, *args, **kwargs):
            raise AssertionError("provider should not run")

    retriever = _bare_retriever(reranker=MustNotRun())
    outcome = retriever.rerank("q", [_hit(1, 0.9, "a"), _hit(2, 0.8, "b")], 1, enabled=False)
    assert outcome.status == "off"
    assert [h.chunk_id for h in outcome.hits] == ["a"]


def test_transient_rerank_failure_falls_back_to_vector_prefix():
    class Unavailable:
        model = "rerank-test"

        def rerank(self, *args, **kwargs):
            error = RuntimeError("secret provider detail")
            error.status_code = 503
            raise error

    retriever = _bare_retriever(reranker=Unavailable())
    outcome = retriever.rerank("q", [_hit(1, 0.9, "a"), _hit(2, 0.8, "b")], 1, enabled=True)
    assert outcome.status == "fallback"
    assert outcome.model == "rerank-test"
    assert [h.chunk_id for h in outcome.hits] == ["a"]
    assert outcome.failure_reason == "provider_unavailable"


def test_non_transient_rerank_failure_is_not_silently_downgraded():
    class Unauthorized:
        def rerank(self, *args, **kwargs):
            error = RuntimeError("invalid key")
            error.status_code = 401
            raise error

    retriever = _bare_retriever(reranker=Unauthorized())
    with pytest.raises(RuntimeError, match="invalid key"):
        retriever.rerank("q", [_hit(1, 0.9, "a")], 1, enabled=True)


@pytest.mark.parametrize(
    ("mode", "override", "expected"),
    [("off", None, False), ("cohere", None, True), ("off", True, True), ("cohere", False, False)],
)
def test_resolve_rerank_enabled(mode, override, expected):
    assert config.resolve_rerank_enabled(mode, override) is expected


def test_resolve_rerank_enabled_rejects_unknown_mode():
    with pytest.raises(ValueError, match="RERANK_MODE"):
        config.resolve_rerank_enabled("mystery", None)


def test_search_attributes_invalid_rerank_mode_to_rerank_stage(monkeypatch):
    retriever = _bare_retriever()
    monkeypatch.setattr(config, "RERANK_MODE", "mystery")

    with pytest.raises(RerankExecutionError) as exc_info:
        retriever.search_with_outcome("q")

    assert isinstance(exc_info.value.__cause__, ValueError)


def test_collections_cover_both_strategies():
    assert set(config.COLLECTIONS) == {"baseline", "structure"}
    assert set(config.CHUNK_PATHS) == {"baseline", "structure"}


def test_to_documents_carries_identity_into_metadata():
    docs, ids = to_documents(_CHUNKS)
    assert len(docs) == len(ids) == 2
    assert docs[0].metadata["chunk_id"] == "article-6"
    assert docs[0].metadata["strategy"] == "structure"
    assert docs[0].metadata["number_int"] == 6
    assert docs[0].page_content == "Article 6 — …"


def test_point_ids_are_deterministic():
    """Same chunk_id must map to the same point id across runs (idempotent upsert)."""
    ids_a = to_documents(_CHUNKS)[1]
    ids_b = to_documents(_CHUNKS)[1]
    assert ids_a == ids_b
    assert len(set(ids_a)) == 2  # distinct chunk_ids -> distinct ids


def test_build_filter_none_when_no_constraints():
    assert build_filter() is None


def test_build_filter_unit_type_only():
    f = build_filter(unit_type="article")
    assert len(f.must) == 1
    cond = f.must[0]
    assert cond.key == "metadata.unit_type"
    assert cond.match.value == "article"


def test_build_filter_number_range_only():
    f = build_filter(number_min=6, number_max=15)
    assert len(f.must) == 1
    cond = f.must[0]
    assert cond.key == "metadata.number_int"
    assert cond.range.gte == 6 and cond.range.lte == 15


def test_build_filter_combines_unit_type_and_range():
    f = build_filter(unit_type="article", number_min=6)
    keys = {c.key for c in f.must}
    assert keys == {"metadata.unit_type", "metadata.number_int"}


def test_score_semantics_guard_passes_on_calibrated_distance():
    # Default config (Cosine) is what the thresholds were calibrated for -> no raise.
    assert config.DISTANCE == config.SCORE_CALIBRATED_DISTANCE
    config.assert_score_threshold_semantics()


def test_score_semantics_guard_raises_on_other_distance(monkeypatch):
    # The whole point: a DISTANCE change must fail loudly (score direction/scale flips),
    # never silently invert the gates that all assume higher-is-better Cosine scores.
    monkeypatch.setattr(config, "DISTANCE", "Euclid")
    with pytest.raises(RuntimeError, match="calibrated for DISTANCE"):
        config.assert_score_threshold_semantics()
