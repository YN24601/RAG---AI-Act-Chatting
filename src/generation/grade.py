"""Two-layer relevance grading: a cheap deterministic score gate, then an
optional LLM semantic check (CRAG/self-RAG style).

score_gate is a pure function (no network) and unit-tested in isolation.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from retrieval import config as retrieval_config
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
    # `>= min_score` assumes a higher-is-better, Cosine-calibrated score; guard so a
    # change to config.DISTANCE fails loudly instead of silently inverting the gate.
    retrieval_config.assert_score_threshold_semantics()
    return hits[0].score >= min_score


def select_answer_hits(
    hits: List[Hit],
    min_score: float = config.ANSWER_MIN_SCORE,
    rel_drop: float = config.ANSWER_REL_DROP,
) -> List[Hit]:
    """Trim the recalled hits to the ones worth grounding an answer on.

    The score gate only vouches for the top hit; the tail can be much weaker and
    dilutes the context. Qdrant returns hits in descending score order, so we keep
    the contiguous prefix at/above a floor that combines an absolute minimum and a
    relative band below the top hit. The top hit is always kept.

    Pure function (no network) — unit-tested in isolation.
    """
    if not hits:
        return []
    # Floor/rel-drop assume a higher-is-better, Cosine-calibrated score; guard so a
    # change to config.DISTANCE fails loudly instead of silently inverting selection.
    retrieval_config.assert_score_threshold_semantics()
    floor = max(min_score, hits[0].score - rel_drop)
    kept = [h for h in hits if h.score >= floor]
    return kept or hits[:1]


def llm_grade(question: str, hits: List[Hit]) -> GradeResult:
    """Ask the LLM whether the retrieved excerpts truly answer the question."""
    grader = get_chat_llm().with_structured_output(GradeResult)
    chain = GRADE_PROMPT | grader
    return chain.invoke({"question": question, "context": format_context(hits)})
