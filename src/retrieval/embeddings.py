"""Mistral embedding model factory (cached)."""
from __future__ import annotations

from functools import lru_cache

from langchain_mistralai import MistralAIEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> MistralAIEmbeddings:
    """A shared MistralAIEmbeddings instance (mistral-embed, 1024-dim).

    langchain-mistralai batches embed_documents internally and retries on 429.
    """
    return MistralAIEmbeddings(model=config.EMBED_MODEL, api_key=config.require("MISTRAL_API_KEY"))
