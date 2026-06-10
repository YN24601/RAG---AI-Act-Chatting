"""Vector retrieval over a Qdrant collection, with a reserved rerank slot.

Two-stage design (per the project plan): vector recall top-k -> (future) rerank
top-n. v1 ships vector-only; the rerank step is an explicit identity passthrough
so the "context precision before/after rerank" delta can be measured later
(Cohere key is present in .env but deferred on purpose).

Supports metadata pre-filtering (unit_type + article-number range) and an
optional min_score gate (the hook Day 5's grade->refuse step will use).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from . import config
from .embeddings import get_embeddings


@dataclass
class Hit:
    rank: int
    score: float
    chunk_id: str
    text: str
    metadata: dict


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
    def __init__(self, strategy: str = "structure", k: int = config.DEFAULT_K):
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

    def _rerank(self, query: str, scored: List, top_n: int) -> List:
        # TODO(rerank): plug Cohere Rerank here (COHERE_API_KEY ready in .env).
        # For now: identity passthrough preserving vector order, truncated to top_n.
        return scored[:top_n]

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        top_n: Optional[int] = None,
        unit_type: Optional[str] = None,
        number_min: Optional[int] = None,
        number_max: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Hit]:
        k = k or self.k
        top_n = top_n or config.DEFAULT_TOP_N
        qfilter = build_filter(unit_type, number_min, number_max)
        scored = self.vs.similarity_search_with_score(query, k=k, filter=qfilter)
        if min_score is not None:
            scored = [(doc, s) for doc, s in scored if s >= min_score]
        scored = self._rerank(query, scored, top_n)
        return [
            Hit(
                rank=i + 1,
                score=round(float(score), 4),
                chunk_id=doc.metadata.get("chunk_id", ""),
                text=doc.page_content,
                metadata=doc.metadata,
            )
            for i, (doc, score) in enumerate(scored)
        ]
