"""Index chunk sets into Qdrant Cloud (one collection per chunking strategy).

Idempotent: a collection whose point count already matches the chunk file is
left untouched (avoids re-paying for embeddings); use recreate=True to rebuild.
"""
from __future__ import annotations

import json
import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

# Payload fields we filter on must be indexed in Qdrant (keyword/integer).
# langchain-qdrant nests chunk metadata under the "metadata" payload key.
_PAYLOAD_INDEXES = {
    "metadata.unit_type": models.PayloadSchemaType.KEYWORD,
    "metadata.strategy": models.PayloadSchemaType.KEYWORD,
    "metadata.number_int": models.PayloadSchemaType.INTEGER,
}

from . import config
from .embeddings import get_embeddings

# Fixed namespace -> deterministic point ids, so re-indexing upserts (no dupes).
_NS = uuid.UUID("a1b2c3d4-0000-4000-8000-000000000000")


def get_client() -> QdrantClient:
    return QdrantClient(
        url=config.require("QDRANT_URL"),
        api_key=config.require("QDRANT_API_KEY"),
        timeout=120,
    )


def load_chunks(strategy: str) -> List[dict]:
    path = config.CHUNK_PATHS[strategy]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/run_ingestion.py first")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_documents(chunks: List[dict]) -> Tuple[List[Document], List[str]]:
    """Convert chunk dicts to LangChain Documents + deterministic point ids."""
    docs, ids = [], []
    for c in chunks:
        meta = dict(c["metadata"])
        meta["chunk_id"] = c["chunk_id"]
        meta["strategy"] = c["strategy"]
        docs.append(Document(page_content=c["text"], metadata=meta))
        ids.append(str(uuid.uuid5(_NS, c["chunk_id"])))
    return docs, ids


def build_collection(strategy: str, recreate: bool = False) -> None:
    name = config.COLLECTIONS[strategy]
    chunks = load_chunks(strategy)
    client = get_client()

    if client.collection_exists(name) and not recreate:
        count = client.count(name).count
        if count == len(chunks):
            print(f"[index] '{name}' already has {count} points — skipping (use --recreate to rebuild)")
            return
        print(f"[index] '{name}' count {count} != {len(chunks)} chunks — rebuilding")

    docs, ids = to_documents(chunks)
    print(f"[index] embedding {len(docs)} '{strategy}' chunks via {config.EMBED_MODEL} -> '{name}' …")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        ids=ids,
        collection_name=name,
        url=config.require("QDRANT_URL"),
        api_key=config.require("QDRANT_API_KEY"),
        force_recreate=True,
        timeout=120,
    )
    _ensure_payload_indexes(get_client(), name)
    final = get_client().count(name).count
    print(f"[index] '{name}' built: {final} points (payload indexes: {', '.join(_PAYLOAD_INDEXES)})")


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """Create the payload indexes needed for metadata filtering (idempotent)."""
    for field, schema in _PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)
        except Exception as exc:  # already exists / race -> safe to ignore
            print(f"[index]   payload index {field}: {exc}")


def build_all(strategies=("baseline", "structure"), recreate: bool = False) -> None:
    for s in strategies:
        build_collection(s, recreate=recreate)
