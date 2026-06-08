"""Network-free unit tests for the retrieval layer (no Qdrant/Mistral calls)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval import config  # noqa: E402
from retrieval.index import to_documents  # noqa: E402

_CHUNKS = [
    {"chunk_id": "article-6", "text": "Article 6 — …", "strategy": "structure",
     "metadata": {"unit_type": "article", "number": "6", "number_int": 6}},
    {"chunk_id": "baseline-0001", "text": "some text", "strategy": "baseline",
     "metadata": {"chunk_index": 1}},
]


def test_collections_cover_both_strategies():
    assert set(config.COLLECTIONS) == {"baseline", "structure"}
    assert set(config.CHUNK_PATHS) == {"baseline", "structure"}


def test_to_documents_carries_identity_into_metadata():
    docs, ids = to_documents(_CHUNKS)
    assert len(docs) == len(ids) == 2
    assert docs[0].metadata["chunk_id"] == "article-6"
    assert docs[0].metadata["strategy"] == "structure"
    assert docs[0].metadata["number_int"] == 6
    assert docs[0].page_content == "Article 6 — …"


def test_point_ids_are_deterministic():
    """Same chunk_id must map to the same point id across runs (idempotent upsert)."""
    ids_a = to_documents(_CHUNKS)[1]
    ids_b = to_documents(_CHUNKS)[1]
    assert ids_a == ids_b
    assert len(set(ids_a)) == 2  # distinct chunk_ids -> distinct ids
