"""Index chunk sets into Qdrant Cloud (one collection per chunking strategy).

Idempotent via a content fingerprint (sha256 of the chunk file): a collection
is rebuilt only when the chunk content actually changed (not just when the count
differs), so editing chunking without changing the count can't leave stale
vectors behind. Use recreate=True to force a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from ingestion.schema import DATA_PROCESSED

from . import config
from .embeddings import get_embeddings

# Payload fields we filter on must be indexed in Qdrant (keyword/integer).
# langchain-qdrant nests chunk metadata under the "metadata" payload key.
_PAYLOAD_INDEXES = {
    "metadata.unit_type": models.PayloadSchemaType.KEYWORD,
    "metadata.number_int": models.PayloadSchemaType.INTEGER,
}

# Map the config distance name to the qdrant enum (config is authoritative).
_DISTANCE = {
    "Cosine": models.Distance.COSINE,
    "Dot": models.Distance.DOT,
    "Euclid": models.Distance.EUCLID,
}

# Fixed namespace -> deterministic point ids, so re-indexing upserts (no dupes).
_NS = uuid.UUID("a1b2c3d4-0000-4000-8000-000000000000")

# Tracks {collection: {"fingerprint": sha256, "count": n}} for idempotency.
_META_PATH = DATA_PROCESSED / ".index_meta.json"


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


def _fingerprint(strategy: str) -> str:
    return hashlib.sha256(config.CHUNK_PATHS[strategy].read_bytes()).hexdigest()


def _load_meta() -> dict:
    if _META_PATH.exists():
        return json.loads(_META_PATH.read_text(encoding="utf-8"))
    return {}


def _save_meta(meta: dict) -> None:
    _META_PATH.parent.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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


def _ensure_payload_indexes(client: QdrantClient, name: str) -> None:
    """Create the payload indexes needed for metadata filtering.

    create_payload_index is idempotent in Qdrant, so a real exception here is a
    genuine error (e.g. a mistyped field) and must surface, not be swallowed.
    """
    for field, schema in _PAYLOAD_INDEXES.items():
        client.create_payload_index(collection_name=name, field_name=field, field_schema=schema)


def build_collection(strategy: str, recreate: bool = False) -> None:
    name = config.COLLECTIONS[strategy]
    chunks = load_chunks(strategy)
    fingerprint = _fingerprint(strategy)
    client = get_client()
    meta = _load_meta()

    if not recreate and client.collection_exists(name):
        record = meta.get(name, {})
        count = client.count(name).count
        if record.get("fingerprint") == fingerprint and count == len(chunks):
            print(f"[index] '{name}' up to date ({count} pts, fingerprint match) — skipping")
            return
        reason = "fingerprint changed" if record.get("fingerprint") != fingerprint else f"count {count}!={len(chunks)}"
        print(f"[index] '{name}' stale ({reason}) — rebuilding")

    # Validate embedding dimension against config (config is the contract).
    probe_dim = len(get_embeddings().embed_query("dimension probe"))
    if probe_dim != config.EMBED_DIM:
        raise RuntimeError(f"{config.EMBED_MODEL} returned dim {probe_dim}, config.EMBED_DIM={config.EMBED_DIM}")

    docs, ids = to_documents(chunks)
    print(f"[index] embedding {len(docs)} '{strategy}' chunks via {config.EMBED_MODEL} -> '{name}' …")
    QdrantVectorStore.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        ids=ids,
        collection_name=name,
        url=config.require("QDRANT_URL"),
        api_key=config.require("QDRANT_API_KEY"),
        distance=_DISTANCE[config.DISTANCE],  # config-driven, no longer inferred
        force_recreate=True,
        timeout=120,
    )
    _ensure_payload_indexes(client, name)

    meta[name] = {"fingerprint": fingerprint, "count": len(chunks)}
    _save_meta(meta)
    final = client.count(name).count
    print(f"[index] '{name}' built: {final} pts, dim={config.EMBED_DIM}, distance={config.DISTANCE}")


def build_all(strategies=("baseline", "structure"), recreate: bool = False) -> None:
    for s in strategies:
        build_collection(s, recreate=recreate)
