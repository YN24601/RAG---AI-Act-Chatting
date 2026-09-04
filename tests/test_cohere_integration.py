"""Opt-in live Cohere smoke test.

Run with ``RUN_COHERE_INTEGRATION=1 pytest tests/test_cohere_integration.py``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import config  # noqa: E402
from retrieval.reranker import CohereReranker  # noqa: E402
from retrieval.retriever import Hit  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_COHERE_INTEGRATION") != "1" or not config.COHERE_API_KEY,
    reason="set RUN_COHERE_INTEGRATION=1 and COHERE_API_KEY for the live smoke test",
)


def test_live_cohere_rerank_returns_bounded_score_and_expected_document():
    hits = [
        Hit(1, 0.8, "cake", "How to bake a cake.", {}, retrieval_rank=1),
        Hit(
            2,
            0.7,
            "article-5",
            "Article 5 prohibits certain manipulative AI practices.",
            {"context_header": "Article 5"},
            retrieval_rank=2,
        ),
    ]
    outcome = CohereReranker(
        api_key=config.COHERE_API_KEY,
        model=config.RERANK_MODEL,
        timeout_s=config.RERANK_TIMEOUT_S,
    ).rerank("Which AI practices are prohibited?", hits, top_n=1)

    assert outcome.hits[0].chunk_id == "article-5"
    assert 0.0 <= outcome.hits[0].rerank_score <= 1.0
