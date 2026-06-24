"""Pydantic v2 request/response models for the serving layer.

Mirrors the RAGState contract from generation.graph and the Hit shape from
retrieval.retriever, so the front-end can render Article/Recital attribution with
zero post-processing. Reuses the `Strategy` Literal already defined for ingestion.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ingestion.schema import Strategy


# --- /ask ---
class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question about the EU AI Act")
    strategy: Strategy = "structure"
    show_context: bool = Field(default=False, description="Include each source's raw chunk text")


class Source(BaseModel):
    """One retrieved provision, surfaced for attribution. Fields come straight
    from Hit.metadata; baseline-strategy chunks carry minimal metadata, so the
    citation/chapter/unit_type fields are optional."""

    rank: int
    score: float
    citation: Optional[str] = None  # = metadata.context_header, e.g. "Article 6 — Classification ..."
    chapter: Optional[str] = None
    unit_type: Optional[str] = None
    chunk_id: str
    used: bool  # did this hit survive select_answer_hits and ground the answer?
    text: Optional[str] = None  # populated only when show_context=true


class AskResponse(BaseModel):
    answer: str
    refused: bool
    grade: Optional[str] = None
    grade_reason: Optional[str] = None
    used_hits: int = 0
    sources: List[Source] = Field(default_factory=list)


# --- /query (pure retrieval; debug/compare) ---
class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    strategy: Strategy = "structure"
    k: Optional[int] = Field(default=None, ge=1, description="Vector recall depth")
    top_n: Optional[int] = Field(default=None, ge=1, description="Hits returned after (future) rerank")
    unit_type: Optional[str] = None
    number_min: Optional[int] = None
    number_max: Optional[int] = None
    min_score: Optional[float] = None


class QueryHit(BaseModel):
    rank: int
    score: float
    citation: Optional[str] = None
    chapter: Optional[str] = None
    unit_type: Optional[str] = None
    chunk_id: str
    text: str


class QueryResponse(BaseModel):
    hits: List[QueryHit] = Field(default_factory=list)


# --- /health ---
class HealthResponse(BaseModel):
    status: str = "ok"
    ready: Optional[bool] = None  # set only when readiness was probed (?ready=1)


# --- errors ---
class ErrorResponse(BaseModel):
    error: str
    stage: Optional[str] = None  # set for PipelineError (which pipeline stage failed)
    detail: Optional[str] = None
