"""LangGraph orchestration for recall, rerank, grading, and grounded generation.

The grade node decides the branch: relevant context -> grounded generation that
cites provisions; irrelevant/empty -> a deterministic refusal (never fabricate
legal text). When LANGSMITH_TRACING is set, the whole graph + every LLM call is
auto-traced; the retrieval step (an off-chain call) is wrapped with @traceable
so the recalled docs + scores show up as a child run.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from retrieval import config as retrieval_config
from retrieval.reranker import RerankOutcome
from retrieval.retriever import Hit, Retriever

from . import config
from .errors import PipelineError
from .grade import llm_grade, score_gate, select_answer_hits
from .llm import get_chat_llm
from .prompts import ANSWER_PROMPT, format_context


class RAGState(TypedDict, total=False):
    question: str
    strategy: str
    rerank: Optional[bool]
    candidates: List[Hit]
    hits: List[Hit]
    answer_hits: List[Hit]
    score_gate_passed: bool
    rerank_status: str
    rerank_model: Optional[str]
    rerank_latency_ms: float
    rerank_failure_reason: Optional[str]
    grade: str  # "relevant" | "irrelevant"
    grade_reason: str
    answer: str
    refused: bool
    used_hits: int  # how many of the recalled hits actually grounded the answer


@lru_cache(maxsize=2)  # one client per strategy (baseline/structure)
def _get_retriever(strategy: str) -> Retriever:
    return Retriever(strategy=strategy)


@traceable(name="retrieve", run_type="retriever")
def _run_recall(question: str, strategy: str) -> List[Hit]:
    """Recall top-k vector candidates; reranking is a separate traced operation."""
    return _get_retriever(strategy).recall(question)


@traceable(name="rerank", run_type="retriever")
def _run_rerank(
    question: str,
    strategy: str,
    candidates: List[Hit],
    override: Optional[bool],
) -> RerankOutcome:
    enabled = retrieval_config.resolve_rerank_enabled(override=override)
    return _get_retriever(strategy).rerank(
        question,
        candidates,
        top_n=config.ANSWER_TOP_N,
        enabled=enabled,
    )


# --- Nodes ---
def retrieve(state: RAGState) -> RAGState:
    # Bare Qdrant call: a timeout/5xx here is a hard failure (no context -> no
    # answer). Wrap it as a controlled PipelineError so the API layer can return a
    # unified error instead of leaking the vector-store client's stack trace.
    try:
        candidates = _run_recall(state["question"], state.get("strategy", "structure"))
    except Exception as e:  # noqa: BLE001 — boundary: any client error -> controlled type
        raise PipelineError("retrieve", "vector store is unavailable") from e
    return {"candidates": candidates}


def apply_score_gate(state: RAGState) -> RAGState:
    """Run the calibrated Qdrant score gate before spending a rerank call."""
    candidates = state.get("candidates", [])
    if not score_gate(candidates):
        top = candidates[0].score if candidates else None
        return {
            "score_gate_passed": False,
            "grade": "irrelevant",
            "grade_reason": f"score gate failed (top={top})",
            "hits": candidates[: config.ANSWER_TOP_N],
            "rerank_status": "off",
            "rerank_model": None,
            "rerank_latency_ms": 0.0,
        }
    return {"score_gate_passed": True}


def rerank(state: RAGState) -> RAGState:
    """Apply second-stage ranking; transient provider errors already fall back."""
    try:
        outcome = _run_rerank(
            state["question"],
            state.get("strategy", "structure"),
            state.get("candidates", []),
            state.get("rerank"),
        )
    except Exception as exc:  # noqa: BLE001 — boundary: controlled API error
        raise PipelineError("rerank", "reranker configuration is invalid or unavailable") from exc
    return {
        "hits": outcome.hits,
        "rerank_status": outcome.status,
        "rerank_model": outcome.model,
        "rerank_latency_ms": outcome.latency_ms,
        "rerank_failure_reason": outcome.failure_reason,
    }


def grade(state: RAGState) -> RAGState:
    hits = state.get("hits", [])
    if not hits:
        return {"grade": "irrelevant", "grade_reason": "no context after reranking"}
    if config.GRADE_USE_LLM:
        # The LLM grader only refines a result the deterministic score gate already
        # passed. If Mistral is unavailable, degrade gracefully to "score gate
        # passed -> relevant" rather than crash the whole request on a soft check.
        try:
            result = llm_grade(state["question"], hits)
        except Exception as e:  # noqa: BLE001 — soft check: fall back, don't fail
            return {
                "grade": "relevant",
                "grade_reason": f"llm grader unavailable ({type(e).__name__}), fell back to score gate",
            }
        return {
            "grade": "relevant" if result.relevant else "irrelevant",
            "grade_reason": result.reason,
        }
    return {"grade": "relevant", "grade_reason": "passed score gate"}


def finalize_answer(raw: str) -> tuple[str, bool]:
    """Map raw generation output to (answer, refused).

    The answer model emits INSUFFICIENT_SENTINEL when the context can't support an
    answer. We map that to the canonical (verbatim) REFUSAL_TEXT and refused=True,
    so an in-generation refusal is guaranteed word-for-word and correctly flagged
    — never an LLM-paraphrased refusal silently labeled as a successful answer.
    """
    if config.INSUFFICIENT_SENTINEL in raw.upper():
        return config.REFUSAL_TEXT, True
    return raw, False


def generate(state: RAGState) -> RAGState:
    used = select_grounding_hits(state["hits"], state.get("rerank_status", "off"))
    # Bare Mistral call: a timeout/5xx here is a hard failure (we have context but
    # can't produce the grounded answer). Wrap as PipelineError — never fabricate,
    # and don't mislabel an outage as a refusal (refused must stay authoritative).
    try:
        chain = ANSWER_PROMPT | get_chat_llm() | StrOutputParser()
        raw = chain.invoke(
            {"question": state["question"], "context": format_context(used)}
        )
    except Exception as e:  # noqa: BLE001 — boundary: any client error -> controlled type
        raise PipelineError("generate", "answer model is unavailable") from e
    answer, refused = finalize_answer(raw)
    return {
        "answer": answer,
        "refused": refused,
        "answer_hits": [] if refused else used,
        "used_hits": 0 if refused else len(used),
    }


def refuse(state: RAGState) -> RAGState:
    return {"answer": config.REFUSAL_TEXT, "refused": True, "answer_hits": [], "used_hits": 0}


def _route(state: RAGState) -> str:
    return "generate" if state["grade"] == "relevant" else "refuse"


def _route_after_score_gate(state: RAGState) -> str:
    return "rerank" if state["score_gate_passed"] else "refuse"


def select_grounding_hits(hits: List[Hit], rerank_status: str) -> List[Hit]:
    """Select the exact contexts passed to generation for each ranking path."""
    if rerank_status == "applied":
        return list(hits)
    return select_answer_hits(hits)


@lru_cache(maxsize=1)
def build_graph():
    """Compile recall -> score gate -> rerank -> semantic grade -> answer/refuse."""
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("score_gate", apply_score_gate)
    g.add_node("rerank", rerank)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("refuse", refuse)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "score_gate")
    g.add_conditional_edges(
        "score_gate",
        _route_after_score_gate,
        {"rerank": "rerank", "refuse": "refuse"},
    )
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", _route, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def answer_question(
    question: str,
    strategy: str = "structure",
    rerank: Optional[bool] = None,
) -> RAGState:
    """Run the full pipeline end-to-end and return the final state."""
    return build_graph().invoke({"question": question, "strategy": strategy, "rerank": rerank})
