"""Central settings for the retrieval stack (loaded from .env)."""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from ingestion.schema import (
    CHUNKS_BASELINE_PATH,
    CHUNKS_STRUCTURE_PATH,
    PROJECT_ROOT,
)

load_dotenv(PROJECT_ROOT / ".env")

# --- Secrets / endpoints ---
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")

# --- Model / vector params ---
EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024  # mistral-embed output dimension
DISTANCE = "Cosine"

# --- Score-threshold semantics (single source of truth) ---
# Qdrant's returned "score" means different things per distance metric:
#   Cosine / Dot -> similarity: higher = more relevant
#   Euclid       -> distance:   lower  = more relevant
# Every score gate in the stack — Retriever.search's `min_score`, grade.score_gate's
# GRADE_MIN_SCORE, and generation's ANSWER_MIN_SCORE/REL_DROP — assumes higher-is-better
# AND uses numeric thresholds empirically calibrated on Cosine during the Day 3-4 probe
# (in-scope ~0.72+, out-of-scope ~0.62). Changing DISTANCE flips both the *direction*
# and the *scale* of the score, which would silently invert/invalidate every gate with
# no error. Pin the calibrated metric here and assert it wherever a threshold is applied,
# so a DISTANCE change fails loudly until the gates are recalibrated for the new metric.
SCORE_CALIBRATED_DISTANCE = "Cosine"


def assert_score_threshold_semantics() -> None:
    """Guard: score gates were calibrated for SCORE_CALIBRATED_DISTANCE.

    Raise (instead of silently mis-judging relevance) if config.DISTANCE has moved to a
    metric whose score direction and threshold magnitudes haven't been re-derived. Call
    this at every site that compares a Qdrant score against a fixed threshold.
    """
    if DISTANCE != SCORE_CALIBRATED_DISTANCE:
        raise RuntimeError(
            f"Score thresholds (min_score / GRADE_MIN_SCORE / ANSWER_MIN_SCORE) were "
            f"calibrated for DISTANCE={SCORE_CALIBRATED_DISTANCE!r} (higher = more relevant), "
            f"but config.DISTANCE is {DISTANCE!r}. Qdrant's score direction and scale change "
            f"with the metric, so the gates would silently mis-judge relevance. Recalibrate "
            f"those thresholds for {DISTANCE!r} (and flip the comparison direction if it is a "
            f"distance metric), then update SCORE_CALIBRATED_DISTANCE — or keep "
            f"DISTANCE={SCORE_CALIBRATED_DISTANCE!r}."
        )

# --- Collections: one per chunking strategy, so we can compare retrieval ---
COLLECTIONS = {"baseline": "aiact_baseline", "structure": "aiact_structure"}
CHUNK_PATHS = {"baseline": CHUNKS_BASELINE_PATH, "structure": CHUNKS_STRUCTURE_PATH}

# --- Retrieval defaults ---
DEFAULT_K = 20  # vector recall depth
DEFAULT_TOP_N = 5  # returned after second-stage reranking

# --- Reranking ---
RERANK_MODE = os.environ.get("RERANK_MODE", "off").strip().lower()
RERANK_MODEL = os.environ.get("RERANK_MODEL", "rerank-v4.0-pro").strip()
RERANK_TIMEOUT_S = float(os.environ.get("RERANK_TIMEOUT_S", "3.0"))


def resolve_rerank_enabled(mode: Optional[str] = None, override: Optional[bool] = None) -> bool:
    """Resolve the deployment mode plus an explicit debug/evaluation override."""
    if override is not None:
        return override
    normalized = (RERANK_MODE if mode is None else mode).strip().lower()
    if normalized not in {"off", "cohere"}:
        raise ValueError("RERANK_MODE must be either 'off' or 'cohere'")
    return normalized == "cohere"


def require(name: str) -> str:
    """Fetch a required env var or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — add it to .env (see .env.example)")
    return value
