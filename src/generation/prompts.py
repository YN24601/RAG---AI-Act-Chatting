"""Prompts + context formatting for grounded generation and LLM grading.

`format_context` is a pure function (no network) so it can be unit-tested and
reused by both the answer prompt and the grade prompt.
"""
from __future__ import annotations

from typing import List

from langchain_core.prompts import ChatPromptTemplate

from retrieval.retriever import Hit

from .config import INSUFFICIENT_SENTINEL


def format_context(hits: List[Hit]) -> str:
    """Render retrieved Hits into a citable, numbered context block.

    Each block leads with the chunk's self-describing header (e.g.
    "Article 5 — Prohibited AI practices") and chapter, so the model can cite a
    concrete provision rather than a faceless snippet.
    """
    if not hits:
        return "(no context retrieved)"
    blocks = []
    for h in hits:
        m = h.metadata
        header = m.get("context_header") or f"chunk {m.get('chunk_index')}"
        chapter = f" [{m['chapter']}]" if m.get("chapter") else ""
        blocks.append(f"[{h.rank}] {header}{chapter}\n{h.text.strip()}")
    return "\n\n".join(blocks)


# --- Grounded answer prompt: only the provided context, cite provisions, else refuse ---
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful assistant answering questions about the EU AI Act "
            "(Regulation (EU) 2024/1689) STRICTLY from the provided excerpts.\n"
            "Rules you must follow:\n"
            "1. Use ONLY the information in the CONTEXT below. Do not rely on outside "
            "knowledge and never invent or paraphrase legal text that is not present.\n"
            "2. For every statement, cite the specific provision it comes from using the "
            "header of the excerpt (e.g. \"Article 5\", \"Annex III\", \"Recital 28\").\n"
            "3. Binding obligations live in the Articles/Annexes; Recitals are non-binding "
            "context. If only Recitals address the question, you may still answer from them, "
            "but state explicitly that this is recital/contextual material, not a binding "
            "definition or obligation.\n"
            "4. Only when the CONTEXT is genuinely off-topic or lacks any basis to answer, "
            "output EXACTLY this token and nothing else:\n"
            f"{INSUFFICIENT_SENTINEL}\n"
            "Be precise and concise.",
        ),
        ("human", "QUESTION:\n{question}\n\nCONTEXT:\n{context}"),
    ]
)


# --- Binary relevance grade: do these excerpts actually answer the question? ---
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a relevance grader for a legal retrieval system. Given a user "
            "QUESTION and retrieved EU AI Act excerpts, decide whether the excerpts "
            "contain information that can actually answer the question. Judge relevance "
            "to the asked question, not mere topical overlap. Set relevant=false if the "
            "excerpts are off-topic or only tangentially related.",
        ),
        ("human", "QUESTION:\n{question}\n\nEXCERPTS:\n{context}"),
    ]
)
