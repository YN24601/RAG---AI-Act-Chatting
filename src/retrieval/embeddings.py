"""Mistral embedding model factory (cached)."""
from __future__ import annotations

import os

# MistralAIEmbeddings downloads the Mixtral tokenizer from the HuggingFace Hub (a
# gated repo) purely to size embedding batches under the 16k-token API limit. We
# never use HF Hub (Mistral is API-only), so an unauthenticated request to a gated
# repo is attempted on every init, emitting "unauthenticated requests to the HF
# Hub" before falling back. Forcing offline loads the tokenizer from local cache if
# present, else falls back to a len()-based counter — either way no network call
# and no warning. Must run before huggingface_hub is first imported; setdefault
# lets a user opt back online (with HF_TOKEN) via HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from functools import lru_cache  # noqa: E402

from langchain_mistralai import MistralAIEmbeddings  # noqa: E402

from . import config  # noqa: E402


@lru_cache(maxsize=1)
def get_embeddings() -> MistralAIEmbeddings:
    """A shared MistralAIEmbeddings instance (mistral-embed, 1024-dim).

    langchain-mistralai batches embed_documents internally and retries on 429.
    """
    return MistralAIEmbeddings(model=config.EMBED_MODEL, api_key=config.require("MISTRAL_API_KEY"))
