"""Vector retrieval over a Qdrant collection, with a reserved rerank slot.

Two-stage design (per the project plan): vector recall top-k -> (future) rerank
top-n. v1 ships vector-only; the rerank step is an explicit identity passthrough
so the "context precision before/after rerank" delta can be measured later
(Cohere key is present in .env but deferred on purpose).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document
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

    def _filter(self, unit_type: Optional[str]) -> Optional[models.Filter]:
        if not unit_type:
            return None
        # langchain-qdrant nests chunk metadata under the "metadata" payload key.
        return models.Filter(
            must=[models.FieldCondition(key="metadata.unit_type", match=models.MatchValue(value=unit_type))]
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
    ) -> List[Hit]:
        k = k or self.k
        top_n = top_n or config.DEFAULT_TOP_N
        scored = self.vs.similarity_search_with_score(query, k=k, filter=self._filter(unit_type))
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
