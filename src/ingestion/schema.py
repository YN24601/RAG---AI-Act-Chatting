"""Shared data models and constants for the ingestion pipeline.

Uses pydantic v2 so the same models can be reused by the FastAPI layer later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- Corpus identity (single source of truth; surfaced in README + metadata) ---
SOURCE_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
CELEX = "32024R1689"
VERSION = "OJ L 2024/1689 (base text, Digital Omnibus amendments NOT included)"

# --- Filesystem layout ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RAW_HTML_PATH = DATA_RAW / "eu_ai_act_2024_1689.html"
FETCH_META_PATH = DATA_RAW / "fetch_metadata.json"
UNITS_PATH = DATA_PROCESSED / "units.jsonl"
CHUNKS_BASELINE_PATH = DATA_PROCESSED / "chunks_baseline.jsonl"
CHUNKS_STRUCTURE_PATH = DATA_PROCESSED / "chunks_structure.jsonl"

# Expected structural counts of Regulation (EU) 2024/1689 — used for sanity warnings.
EXPECTED_COUNTS = {"recital": 180, "article": 113, "annex": 13}

UnitType = Literal["recital", "article", "annex"]
Strategy = Literal["baseline", "structure"]


class LegalUnit(BaseModel):
    """One structural unit of the regulation (a recital, an article, or an annex)."""

    unit_id: str  # e.g. "article-6", "recital-14", "annex-III"
    unit_type: UnitType
    number: str  # display form: "6", "14", "III"
    number_int: Optional[int] = None  # numeric form for range filtering/sorting (annex roman -> int)
    title: Optional[str] = None  # article subtitle / annex descriptive title
    chapter: Optional[str] = None  # e.g. "CHAPTER III — HIGH-RISK AI SYSTEMS"
    section: Optional[str] = None  # e.g. "SECTION 1 — Classification ..."
    text: str
    source_url: str = SOURCE_URL
    version: str = VERSION


class Chunk(BaseModel):
    """A retrievable chunk produced by one of the chunking strategies."""

    chunk_id: str
    text: str
    strategy: Strategy
    metadata: dict = Field(default_factory=dict)
