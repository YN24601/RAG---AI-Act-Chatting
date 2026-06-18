"""Two-layer relevance grading: a cheap deterministic score gate, then an
optional LLM semantic check (CRAG/self-RAG style).

score_gate is a pure function (no network) and unit-tested in isolation.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from retrieval.retriever import Hit

from . import config
from .llm import get_chat_llm
from .prompts import GRADE_PROMPT, format_context


class GradeResult(BaseModel):
    """Structured output for the LLM relevance grader."""

    relevant: bool = Field(description="True if the excerpts can answer the question")
    reason: str = Field(description="One short sentence justifying the decision")


def score_gate(hits: List[Hit], min_score: float = config.GRADE_MIN_SCORE) -> bool:
    """Fast, deterministic pre-filter: pass only if the top hit clears min_score.

    Empty recall or a weak top score -> refuse without spending an LLM call.
    """
    if not hits:
        return False
    return hits[0].score >= min_score


def llm_grade(question: str, hits: List[Hit]) -> GradeResult:
    """Ask the LLM whether the retrieved excerpts truly answer the question."""
    grader = get_chat_llm().with_structured_output(GradeResult)
    chain = GRADE_PROMPT | grader
    return chain.invoke({"question": question, "context": format_context(hits)})
