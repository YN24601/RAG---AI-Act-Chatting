"""Two-stage retrieval over Qdrant recall candidates and an optional reranker.

Supports metadata pre-filtering (unit_type + article-number range) and an
optional min_score gate (the hook Day 5's grade->refuse step will use).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from . import config
from .embeddings import get_embeddings
from .reranker import CohereReranker, RerankOutcome, is_transient_rerank_error


class RerankExecutionError(RuntimeError):
    """Marks a non-fallback reranker failure for API stage attribution."""


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    text: str
    metadata: dict
    retrieval_rank: int = 0
    rerank_score: Optional[float] = None


def build_filter(
    unit_type: Optional[str] = None,
    number_min: Optional[int] = None,
    number_max: Optional[int] = None,
) -> Optional[models.Filter]:
    """Build a Qdrant filter from optional metadata constraints (None -> no filter).

    Pure function (no network/state) so it can be unit-tested in isolation.
    langchain-qdrant nests chunk metadata under the "metadata" payload key.
    """
    must: List[models.FieldCondition] = []
    if unit_type:
        must.append(
            models.FieldCondition(key="metadata.unit_type", match=models.MatchValue(value=unit_type))
        )
    if number_min is not None or number_max is not None:
        must.append(
            models.FieldCondition(
                key="metadata.number_int",
                range=models.Range(gte=number_min, lte=number_max),
            )
        )
    return models.Filter(must=must) if must else None


class Retriever:
    def __init__(
        self,
        strategy: str = "structure",
        k: int = config.DEFAULT_K,
        reranker=None,
    ):
        if strategy not in config.COLLECTIONS:
            raise ValueError(f"unknown strategy {strategy!r}; expected one of {list(config.COLLECTIONS)}")
        self.strategy = strategy
        self.k = k
        self.client = QdrantClient(
            url=config.require("QDRANT_URL"),
            api_key=config.require("QDRANT_API_KEY"),
            timeout=60,
        )
        self.vs = QdrantVectorStore(
            client=self.client,
            collection_name=config.COLLECTIONS[strategy],
            embedding=get_embeddings(),
        )
        self._reranker_client = reranker

    def _get_reranker(self):
        if self._reranker_client is None:
            self._reranker_client = CohereReranker(
                api_key=config.COHERE_API_KEY,
                model=config.RERANK_MODEL,
                timeout_s=config.RERANK_TIMEOUT_S,
            )
        return self._reranker_client

    @staticmethod
    def _vector_outcome(
        hits: List[Hit],
        top_n: int,
        status: str,
        model: Optional[str] = None,
        **kwargs,
    ) -> RerankOutcome:
        selected = [replace(hit, rank=i + 1, rerank_score=None) for i, hit in enumerate(hits[:top_n])]
        return RerankOutcome(hits=selected, status=status, model=model, latency_ms=0.0, **kwargs)

    def recall(
        self,
        query: str,
        k: Optional[int] = None,
        unit_type: Optional[str] = None,
        number_min: Optional[int] = None,
        number_max: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Hit]:
        """Recall vector candidates without invoking the second-stage reranker."""
        k = self.k if k is None else k
        qfilter = build_filter(unit_type, number_min, number_max)
        scored = self.vs.similarity_search_with_score(query, k=k, filter=qfilter)
        if min_score is not None:
            config.assert_score_threshold_semantics()
            scored = [(doc, score) for doc, score in scored if score >= min_score]
        return [
            Hit(
                rank=i + 1,
                retrieval_rank=i + 1,
                score=round(float(score), 4),
                rerank_score=None,
                chunk_id=doc.metadata.get("chunk_id", ""),
                text=doc.page_content,
                metadata=doc.metadata,
            )
            for i, (doc, score) in enumerate(scored)
        ]

    def rerank(
        self,
        query: str,
        candidates: List[Hit],
        top_n: int,
        enabled: bool,
    ) -> RerankOutcome:
        """Rerank candidates or return a vector-only/fallback outcome."""
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not enabled:
            return self._vector_outcome(candidates, top_n, "off")
        started = time.perf_counter()
        try:
            return self._get_reranker().rerank(query, candidates, top_n)
        except Exception as exc:
            if not is_transient_rerank_error(exc):
                raise
            outcome = self._vector_outcome(
                candidates,
                top_n,
                "fallback",
                model=getattr(self._reranker_client, "model", config.RERANK_MODEL),
                failure_reason="provider_unavailable",
            )
            outcome.latency_ms = round((time.perf_counter() - started) * 1000, 2)
            return outcome

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        unit_type: Optional[str] = None,
        number_min: Optional[int] = None,
        number_max: Optional[int] = None,
        min_score: Optional[float] = None,
        rerank: Optional[bool] = None,
    ) -> List[Hit]:
        return self.search_with_outcome(
            query,
            k=k,
            top_n=top_n,
            unit_type=unit_type,
            number_min=number_min,
            number_max=number_max,
            min_score=min_score,
            rerank=rerank,
        ).hits

    def search_with_outcome(
        self,
        query: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        unit_type: Optional[str] = None,
        number_min: Optional[int] = None,
        number_max: Optional[int] = None,
        min_score: Optional[float] = None,
        rerank: Optional[bool] = None,
    ) -> RerankOutcome:
        """Run both retrieval stages while preserving rerank status metadata."""
        top_n = config.DEFAULT_TOP_N if top_n is None else top_n
        try:
            enabled = config.resolve_rerank_enabled(override=rerank)
        except Exception as exc:
            raise RerankExecutionError("invalid reranker configuration") from exc
        candidates = self.recall(
            query,
            k=k,
            unit_type=unit_type,
            number_min=number_min,
            number_max=number_max,
            min_score=min_score,
        )
        try:
            return self.rerank(query, candidates, top_n, enabled)
        except Exception as exc:
            raise RerankExecutionError("non-transient reranker failure") from exc
