"""FastAPI app: wrap the Day 5 RAG pipeline as a deployable service.

Endpoints:
  POST /ask     full QA   -> generation.graph.answer_question
  POST /query   retrieval -> retrieval.retriever.Retriever.search (debug/compare)
  GET  /health  liveness; optional ?ready=1 probes Qdrant for readiness
  GET  /        same-origin static front-end (Stage 2; mounted if present)

Error policy (the hook Day 5 reserved): a hard-dependency outage surfaces as a
single controlled `PipelineError`, which maps to HTTP 503 with a clean body and
no leaked stack. Crucially, an outage is NEVER written as `refused` — `refused`
stays the authoritative "the corpus has no basis to answer" signal.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from generation.errors import PipelineError
from generation.graph import _get_retriever, answer_question
from retrieval import config as retrieval_config
from retrieval.retriever import RerankExecutionError

from .schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
    Source,
)

logger = logging.getLogger("api")

app = FastAPI(
    title="EU AI Act QA",
    description="Grounded retrieval-augmented QA over Regulation (EU) 2024/1689.",
    version="0.1.0",
)


# --- Error handlers (outage != refusal) ---
@app.exception_handler(PipelineError)
async def pipeline_error_handler(request: Request, exc: PipelineError) -> JSONResponse:
    # A backing service (Qdrant/Mistral) failed. The original cause is already
    # chained into the exception for the server logs; clients get a clean 503
    # with the failing stage and a caller-safe message — no internal stack.
    logger.exception("PipelineError at stage %s on %s", exc.stage, request.url.path)
    body = ErrorResponse(error="service unavailable", stage=exc.stage, detail=str(exc))
    return JSONResponse(status_code=503, content=body.model_dump())


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    body = ErrorResponse(error="internal server error", detail="an unexpected error occurred")
    return JSONResponse(status_code=500, content=body.model_dump())


# --- Routes ---
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Full pipeline: retrieve -> grade -> generate | refuse."""
    state = answer_question(req.question, req.strategy)
    hits = state.get("hits", []) or []
    refused = bool(state.get("refused", False))

    # Use the graph's authoritative list rather than re-deriving selection from
    # scores (reranked and vector-fallback paths intentionally select differently).
    used_ids = {h.chunk_id for h in state.get("answer_hits", [])} if not refused else set()

    sources = [
        Source(
            rank=h.rank,
            score=h.score,
            retrieval_rank=h.retrieval_rank or h.rank,
            rerank_score=h.rerank_score,
            citation=h.metadata.get("context_header"),
            chapter=h.metadata.get("chapter"),
            unit_type=h.metadata.get("unit_type"),
            chunk_id=h.chunk_id,
            used=h.chunk_id in used_ids,
            text=h.text if req.show_context else None,
        )
        for h in hits
    ]

    return AskResponse(
        answer=state.get("answer", ""),
        refused=refused,
        grade=state.get("grade"),
        grade_reason=state.get("grade_reason"),
        used_hits=state.get("used_hits", 0),
        rerank_status=state.get("rerank_status", "off"),
        rerank_model=state.get("rerank_model"),
        rerank_latency_ms=state.get("rerank_latency_ms", 0.0),
        rerank_failure_reason=state.get("rerank_failure_reason"),
        sources=sources,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Retrieval debugging endpoint with an optional rerank override."""
    retriever = _get_retriever(req.strategy)  # cached per strategy
    try:
        outcome = retriever.search_with_outcome(
            req.question,
            k=req.k,
            top_n=req.top_n,
            unit_type=req.unit_type,
            number_min=req.number_min,
            number_max=req.number_max,
            min_score=req.min_score,
            rerank=req.rerank,
        )
    except RerankExecutionError as e:
        raise PipelineError("rerank", "reranker configuration is invalid or unavailable") from e
    except Exception as e:  # noqa: BLE001 — boundary: surface as a controlled 503
        raise PipelineError("retrieve", "vector store is unavailable") from e

    return QueryResponse(
        hits=[
            QueryHit(
                rank=h.rank,
                score=h.score,
                retrieval_rank=h.retrieval_rank or h.rank,
                rerank_score=h.rerank_score,
                citation=h.metadata.get("context_header"),
                chapter=h.metadata.get("chapter"),
                unit_type=h.metadata.get("unit_type"),
                chunk_id=h.chunk_id,
                text=h.text,
            )
            for h in outcome.hits
        ],
        rerank_status=outcome.status,
        rerank_model=outcome.model,
        rerank_latency_ms=outcome.latency_ms,
        rerank_failure_reason=outcome.failure_reason,
    )


@app.get("/health", response_model=HealthResponse)
def health(ready: bool = False, strategy: str = "structure") -> HealthResponse:
    """Liveness by default. With ?ready=1, best-effort readiness probe of Qdrant
    (a failed probe reports ready=false, never a 5xx — readiness is advisory)."""
    if not ready:
        return HealthResponse(status="ok")
    try:
        client = _get_retriever(strategy).client
        is_ready = client.collection_exists(retrieval_config.COLLECTIONS[strategy])
    except Exception:  # noqa: BLE001 — readiness is best-effort, never fatal
        logger.warning("readiness probe failed", exc_info=True)
        is_ready = False
    return HealthResponse(status="ok", ready=is_ready)


# --- Static front-end (Stage 2): mounted only if the directory exists, so the
# API runs standalone before the page is added. ---
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
