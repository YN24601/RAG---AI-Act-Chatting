"""LangGraph orchestration: retrieve -> grade -> generate | refuse (Day 5).

The grade node decides the branch: relevant context -> grounded generation that
cites provisions; irrelevant/empty -> a deterministic refusal (never fabricate
legal text). When LANGSMITH_TRACING is set, the whole graph + every LLM call is
auto-traced; the retrieval step (an off-chain call) is wrapped with @traceable
so the recalled docs + scores show up as a child run.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from retrieval.retriever import Hit, Retriever

from . import config
from .grade import llm_grade, score_gate
from .llm import get_chat_llm
from .prompts import ANSWER_PROMPT, format_context


class RAGState(TypedDict, total=False):
    question: str
    strategy: str
    hits: List[Hit]
    grade: str  # "relevant" | "irrelevant"
    grade_reason: str
    answer: str
    refused: bool


@lru_cache(maxsize=2)  # one client per strategy (baseline/structure)
def _get_retriever(strategy: str) -> Retriever:
    return Retriever(strategy=strategy)


@traceable(name="retrieve", run_type="retriever")
def _run_retrieval(question: str, strategy: str) -> List[Hit]:
    """Off-chain vector retrieval; @traceable surfaces docs+scores in the trace."""
    return _get_retriever(strategy).search(question, top_n=config.ANSWER_TOP_N)


# --- Nodes ---
def retrieve(state: RAGState) -> RAGState:
    hits = _run_retrieval(state["question"], state.get("strategy", "structure"))
    return {"hits": hits}


def grade(state: RAGState) -> RAGState:
    hits = state["hits"]
    if not score_gate(hits):
        top = hits[0].score if hits else None
        return {"grade": "irrelevant", "grade_reason": f"score gate failed (top={top})"}
    if config.GRADE_USE_LLM:
        result = llm_grade(state["question"], hits)
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
    chain = ANSWER_PROMPT | get_chat_llm() | StrOutputParser()
    raw = chain.invoke(
        {"question": state["question"], "context": format_context(state["hits"])}
    )
    answer, refused = finalize_answer(raw)
    return {"answer": answer, "refused": refused}


def refuse(state: RAGState) -> RAGState:
    return {"answer": config.REFUSAL_TEXT, "refused": True}


def _route(state: RAGState) -> str:
    return "generate" if state["grade"] == "relevant" else "refuse"


@lru_cache(maxsize=1)
def build_graph():
    """Compile the retrieve -> grade -> (generate | refuse) graph (cached)."""
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("refuse", refuse)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    return g.compile()


def answer_question(question: str, strategy: str = "structure") -> RAGState:
    """Run the full pipeline end-to-end and return the final state."""
    return build_graph().invoke({"question": question, "strategy": strategy})
