"""Second-stage reranking boundary for retrieved legal-text candidates.

This module deliberately uses Cohere's SDK directly instead of LangChain's
document-compressor abstraction.  The explicit boundary lets the application
retain Qdrant rank/score provenance while attaching the independent,
query-relative Cohere score.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional, Sequence

import httpx


RerankStatus = Literal["off", "applied", "fallback"]
_TRANSIENT_STATUS_CODES = {429, 500, 503, 504}


class RerankConfigurationError(RuntimeError):
    """Raised when reranking was enabled with unusable configuration."""


class RerankProtocolError(RuntimeError):
    """Raised when a provider response cannot be mapped to input candidates."""


@dataclass
class RerankOutcome:
    hits: list[Any]
    status: RerankStatus
    model: Optional[str]
    latency_ms: float
    failure_reason: Optional[str] = None


def _status_code(exc: BaseException) -> Optional[int]:
    value = getattr(exc, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_transient_rerank_error(exc: BaseException) -> bool:
    """Return whether an error may safely degrade to vector-only ranking."""
    if isinstance(
        exc,
        (TimeoutError, ConnectionError, OSError, httpx.TimeoutException, httpx.NetworkError),
    ):
        return True
    return _status_code(exc) in _TRANSIENT_STATUS_CODES


def _format_candidate(hit: Any) -> str:
    metadata = hit.metadata
    provision = metadata.get("context_header") or f"chunk {metadata.get('chunk_index', '')}"
    chapter = metadata.get("chapter")
    prefix = f"Provision: {provision}"
    if chapter:
        prefix += f"\nChapter: {chapter}"
    return f"{prefix}\nText: {hit.text.strip()}"


class CohereReranker:
    """Thin, injectable adapter over ``cohere.ClientV2.rerank``."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-v4.0-pro",
        timeout_s: float = 3.0,
        client: Any = None,
    ) -> None:
        if not model:
            raise RerankConfigurationError("RERANK_MODEL must not be empty")
        self.model = model
        if client is None:
            if not api_key:
                raise RerankConfigurationError(
                    "COHERE_API_KEY is required when RERANK_MODE='cohere'"
                )
            try:
                import cohere
            except ImportError as exc:
                raise RerankConfigurationError(
                    "the 'cohere' package is required when RERANK_MODE='cohere'"
                ) from exc
            client = cohere.ClientV2(
                api_key=api_key,
                timeout=timeout_s,
                max_retries=0,
                client_name="eu-ai-act-rag",
            )
        self.client = client

    def rerank(self, query: str, candidates: Sequence[Any], top_n: int) -> RerankOutcome:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not candidates:
            return RerankOutcome([], "applied", self.model, 0.0)

        started = time.perf_counter()
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[_format_candidate(hit) for hit in candidates],
            top_n=min(top_n, len(candidates)),
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        expected_count = min(top_n, len(candidates))
        if len(response.results) != expected_count:
            raise RerankProtocolError(
                f"Cohere returned {len(response.results)} results; expected {expected_count}"
            )

        mapped = []
        seen: set[int] = set()
        for final_rank, result in enumerate(response.results, 1):
            index = int(result.index)
            if index < 0 or index >= len(candidates) or index in seen:
                raise RerankProtocolError(
                    f"invalid or duplicate candidate index {index} in Cohere response"
                )
            seen.add(index)
            score = float(result.relevance_score)
            if not 0.0 <= score <= 1.0:
                raise RerankProtocolError(f"invalid Cohere relevance_score {score}")
            mapped.append(replace(candidates[index], rank=final_rank, rerank_score=score))

        return RerankOutcome(mapped, "applied", self.model, latency_ms)
