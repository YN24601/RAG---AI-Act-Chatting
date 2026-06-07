"""Two chunking strategies over the parsed LegalUnits, for retrieval comparison.

  baseline   : structure-blind fixed-size splitting of the flat document text.
               Keeps only source/version/index metadata (the naive control).
  structure  : one chunk per recital/article/annex; long units are sub-split
               while preserving the full unit metadata + a sub_index. Keeps
               clause integrity and article-level traceability (the contender).

tiktoken cl100k_base is used only as a token *sizing* proxy. Mistral's tokenizer
differs, but exact counts don't matter for chunk-size control.
"""
from __future__ import annotations

import json
from functools import lru_cache
from statistics import mean, median
from typing import Callable, List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import (
    CHUNKS_BASELINE_PATH,
    CHUNKS_STRUCTURE_PATH,
    SOURCE_URL,
    VERSION,
    Chunk,
    LegalUnit,
)

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


@lru_cache(maxsize=1)
def _encoder():
    return tiktoken.get_encoding("cl100k_base")


def token_len(text: str) -> int:
    return len(_encoder().encode(text))


def _splitter(chunk_size: int = CHUNK_SIZE) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _context_header(u: LegalUnit) -> str:
    """A short self-contained prefix so a (sub-)chunk's text reveals its origin.

    Prepended to the embedded text so the embedding model — which sees only the
    text, never the metadata — knows which clause a fragment belongs to.
    """
    label = {"recital": "Recital", "article": "Article", "annex": "Annex"}[u.unit_type]
    head = f"{label} {u.number}"
    if u.title:
        head += f" — {u.title}"
    return head


def chunk_baseline(units: List[LegalUnit]) -> List[Chunk]:
    """Structure-blind: flatten everything, then fixed-size split."""
    full_text = "\n\n".join(u.text for u in units)
    pieces = _splitter().split_text(full_text)
    return [
        Chunk(
            chunk_id=f"baseline-{i:04d}",
            text=piece,
            strategy="baseline",
            metadata={"chunk_index": i, "source_url": SOURCE_URL, "version": VERSION},
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_structure(units: List[LegalUnit]) -> List[Chunk]:
    """Structure-aware: 1 chunk per unit; sub-split only oversized units.

    Each chunk's text is prefixed with a context header (e.g. "Article 6 —
    Classification rules…") so fragments stay self-contained for retrieval.
    Token room for the header is reserved so chunks still fit CHUNK_SIZE.
    """
    chunks: List[Chunk] = []
    for u in units:
        header = _context_header(u)
        prefix = f"{header}\n\n"
        budget = max(128, CHUNK_SIZE - token_len(prefix))
        base_meta = {
            "unit_type": u.unit_type,
            "number": u.number,
            "number_int": u.number_int,
            "title": u.title,
            "chapter": u.chapter,
            "section": u.section,
            "context_header": header,
            "source_url": SOURCE_URL,
            "version": VERSION,
        }
        if token_len(u.text) <= budget:
            chunks.append(
                Chunk(
                    chunk_id=u.unit_id,
                    text=prefix + u.text,
                    strategy="structure",
                    metadata={**base_meta, "sub_index": 0},
                )
            )
        else:
            for j, piece in enumerate(_splitter(budget).split_text(u.text)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{u.unit_id}#s{j}",
                        text=prefix + piece,
                        strategy="structure",
                        metadata={**base_meta, "sub_index": j},
                    )
                )
    return chunks


def _write(chunks: List[Chunk], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")


def _stats(chunks: List[Chunk]) -> dict:
    lengths = sorted(token_len(c.text) for c in chunks)
    if not lengths:
        return {"count": 0, "mean": 0, "median": 0, "p95": 0}
    p95 = lengths[min(len(lengths) - 1, int(round(0.95 * (len(lengths) - 1))))]
    return {
        "count": len(lengths),
        "mean": round(mean(lengths), 1),
        "median": round(median(lengths), 1),
        "p95": p95,
    }


def build_and_write(units: List[LegalUnit]) -> dict:
    """Run both strategies, persist them, and return a stats summary."""
    baseline = chunk_baseline(units)
    structure = chunk_structure(units)
    _write(baseline, CHUNKS_BASELINE_PATH)
    _write(structure, CHUNKS_STRUCTURE_PATH)
    print(f"[chunk] wrote {len(baseline)} baseline chunks -> {CHUNKS_BASELINE_PATH}")
    print(f"[chunk] wrote {len(structure)} structure chunks -> {CHUNKS_STRUCTURE_PATH}")
    return {"baseline": _stats(baseline), "structure": _stats(structure)}
