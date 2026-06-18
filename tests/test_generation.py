"""Network-free unit tests for the generation/orchestration layer (Day 5).

No Mistral / Qdrant / LangSmith calls: only pure helpers and graph assembly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generation import config  # noqa: E402
from generation.graph import build_graph, finalize_answer  # noqa: E402
from generation.grade import score_gate  # noqa: E402
from generation.prompts import format_context  # noqa: E402
from retrieval.retriever import Hit  # noqa: E402


def _hit(rank, score, header="Article 5 — Prohibited AI practices", chapter="Chapter II"):
    return Hit(
        rank=rank,
        score=score,
        chunk_id=f"article-5-{rank}",
        text=f"Some provision text {rank}.",
        metadata={"context_header": header, "chapter": chapter, "unit_type": "article"},
    )


def test_score_gate_empty_is_false():
    assert score_gate([]) is False


def test_score_gate_below_threshold_is_false():
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE - 0.05)]) is False


def test_score_gate_at_or_above_threshold_is_true():
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE)]) is True
    assert score_gate([_hit(1, config.GRADE_MIN_SCORE + 0.2)]) is True


def test_format_context_empty():
    assert format_context([]) == "(no context retrieved)"


def test_format_context_carries_headers_in_rank_order():
    ctx = format_context([_hit(1, 0.9, header="Article 5 — Prohibited AI practices"),
                          _hit(2, 0.8, header="Article 6 — Classification rules")])
    assert "Article 5" in ctx and "Article 6" in ctx
    assert "Chapter II" in ctx
    assert ctx.index("[1]") < ctx.index("[2]")  # preserves rank order


def test_refusal_text_is_stable_and_nonempty():
    assert config.REFUSAL_TEXT
    assert "cannot confirm" in config.REFUSAL_TEXT.lower()


def test_finalize_answer_passes_real_answer_through():
    ans, refused = finalize_answer("Under Article 5, the following practices are prohibited...")
    assert refused is False
    assert ans.startswith("Under Article 5")


def test_finalize_answer_maps_sentinel_to_verbatim_refusal():
    # In-generation refusal must be verbatim REFUSAL_TEXT and flagged refused=True,
    # not an LLM paraphrase mislabeled as a successful answer (the bug this fixes).
    for raw in (config.INSUFFICIENT_SENTINEL, f"  {config.INSUFFICIENT_SENTINEL}\n", "insufficient_context"):
        ans, refused = finalize_answer(raw)
        assert refused is True
        assert ans == config.REFUSAL_TEXT


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    assert {"retrieve", "grade", "generate", "refuse"} <= nodes
