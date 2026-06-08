"""Central settings for the retrieval stack (loaded from .env)."""
from __future__ import annotations

import os

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
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")  # reserved for rerank (deferred)

# --- Model / vector params ---
EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024  # mistral-embed output dimension
DISTANCE = "Cosine"

# --- Collections: one per chunking strategy, so we can compare retrieval ---
COLLECTIONS = {"baseline": "aiact_baseline", "structure": "aiact_structure"}
CHUNK_PATHS = {"baseline": CHUNKS_BASELINE_PATH, "structure": CHUNKS_STRUCTURE_PATH}

# --- Retrieval defaults ---
DEFAULT_K = 20  # vector recall depth
DEFAULT_TOP_N = 5  # returned after (future) rerank


def require(name: str) -> str:
    """Fetch a required env var or fail with a clear message."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set — add it to .env (see .env.example)")
    return value
