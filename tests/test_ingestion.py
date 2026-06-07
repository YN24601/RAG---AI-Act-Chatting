"""Sanity checks for the Day 1-2 ingestion outputs.

Run the pipeline first:  python scripts/run_ingestion.py
Then:                     pytest -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.schema import (  # noqa: E402
    CHUNKS_BASELINE_PATH,
    CHUNKS_STRUCTURE_PATH,
    EXPECTED_COUNTS,
    UNITS_PATH,
)


def _load_jsonl(path):
    if not path.exists():
        pytest.skip(f"{path} missing — run scripts/run_ingestion.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_unit_counts_match_expected():
    units = _load_jsonl(UNITS_PATH)
    for unit_type, expected in EXPECTED_COUNTS.items():
        got = sum(1 for u in units if u["unit_type"] == unit_type)
        assert got == expected, f"{unit_type}: got {got}, expected {expected}"


def test_articles_have_number_and_chapter():
    units = _load_jsonl(UNITS_PATH)
    articles = [u for u in units if u["unit_type"] == "article"]
    assert articles
    for a in articles:
        assert a["number"], f"article missing number: {a['unit_id']}"
        assert a["chapter"], f"article missing chapter: {a['unit_id']}"
        assert a["text"].strip(), f"article empty text: {a['unit_id']}"


def test_no_residual_html_in_units():
    units = _load_jsonl(UNITS_PATH)
    for u in units:
        assert "<" not in u["text"] and ">" not in u["text"], f"residual HTML in {u['unit_id']}"


def test_chunk_files_nonempty_and_tagged():
    baseline = _load_jsonl(CHUNKS_BASELINE_PATH)
    structure = _load_jsonl(CHUNKS_STRUCTURE_PATH)
    assert baseline and structure
    assert all(c["strategy"] == "baseline" for c in baseline)
    assert all(c["strategy"] == "structure" for c in structure)


def test_structure_chunks_carry_article_metadata():
    structure = _load_jsonl(CHUNKS_STRUCTURE_PATH)
    article_chunks = [c for c in structure if c["metadata"].get("unit_type") == "article"]
    assert article_chunks
    for c in article_chunks:
        assert c["metadata"].get("number")
        assert c["metadata"].get("chapter")
