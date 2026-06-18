"""Mistral chat model factory (cached), mirroring retrieval.embeddings."""
from __future__ import annotations

from functools import lru_cache

from langchain_mistralai import ChatMistralAI

from retrieval import config as rconfig

from . import config


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatMistralAI:
    """A shared ChatMistralAI instance for grounded generation + LLM grading.

    Temperature 0 for deterministic, grounded legal answers. langchain-mistralai
    handles batching/retries; LangSmith auto-traces every call when env is set.
    """
    return ChatMistralAI(
        model=config.GEN_MODEL,
        temperature=config.GEN_TEMPERATURE,
        api_key=rconfig.require("MISTRAL_API_KEY"),
    )
