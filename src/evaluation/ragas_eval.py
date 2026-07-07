"""RAGAS scoring wired to the project's Mistral judge (Day 8-9).

The four metrics (per the plan): faithfulness, answer/response relevancy, context
precision (with reference), context recall. RAGAS defaults to OpenAI; we override
its LLM + embeddings with the project's Mistral instances to keep the whole stack
European and consistent with what production actually uses.

### The langchain-1.x shim (why the sys.modules hack below exists)

ragas (through 0.4.x) unconditionally does, at import time:
    from langchain_community.chat_models.vertexai import ChatVertexAI
That submodule was REMOVED when the project moved to the langchain-1.0 split
(langchain-community 0.4.x). Every other symbol ragas needs still resolves — this
is the single missing one. ragas only references ChatVertexAI in an
isinstance/"supports multiple completions" table; it never instantiates it for a
Mistral judge. So we register a one-symbol stub module BEFORE importing ragas.

Rejected alternatives: (a) downgrading the project's langchain to the 0.3 line to
satisfy old ragas — would risk the working, deployed retrieve/grade/generate
pipeline and create dev/prod drift; (b) an isolated venv + subprocess just to run
ragas — heavier, and unnecessary once we saw only one symbol is missing. The shim
is local to this module (survives reinstalls) and documented here.
"""
from __future__ import annotations

import sys
import types
from typing import List

from generation.llm import get_chat_llm
from retrieval.embeddings import get_embeddings

from .harness import RunResult

# --- shim: must run before `import ragas` (see module docstring) ---
_VERTEXAI_PATH = "langchain_community.chat_models.vertexai"
if _VERTEXAI_PATH not in sys.modules:
    try:  # if a real (older) langchain-community provides it, don't shadow it
        __import__(_VERTEXAI_PATH)
    except ModuleNotFoundError:
        _shim = types.ModuleType(_VERTEXAI_PATH)

        class ChatVertexAI:  # noqa: D401 — stub; never instantiated with a Mistral judge
            """Placeholder for the removed langchain_community ChatVertexAI symbol."""

        _shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
        sys.modules[_VERTEXAI_PATH] = _shim

# ragas imports are safe only after the shim above.
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402

# The four metric names as they appear in RAGAS's result columns.
METRIC_KEYS = ["faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]


def ragas_version() -> str:
    import ragas

    return ragas.__version__


def _to_samples(answered: List[RunResult]) -> List[SingleTurnSample]:
    """Map answered runs to RAGAS samples (user_input/response/contexts/reference)."""
    return [
        SingleTurnSample(
            user_input=r.item.question,
            response=r.answer,
            retrieved_contexts=r.contexts or [""],  # RAGAS wants a non-empty list
            reference=r.item.ground_truth,
        )
        for r in answered
    ]


def score_answered(answered: List[RunResult], max_workers: int = 2) -> dict:
    """Run the four RAGAS metrics over the answered subset.

    Returns {metric_key: mean_score} (means over the subset), plus `n_scored`.
    `max_workers` is kept low on purpose — Mistral's free tier 429s under RAGAS's
    default concurrency. Returns zeros with n_scored=0 if the subset is empty
    (e.g. a --limit smoke run that only hit refusals).
    """
    if not answered:
        return {k: 0.0 for k in METRIC_KEYS} | {"n_scored": 0}

    llm = LangchainLLMWrapper(get_chat_llm())
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    dataset = EvaluationDataset(samples=_to_samples(answered))
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            # strictness=1 on purpose: the default (3) asks the judge for 3 completions
            # in one call, which trips a bug in langchain-mistralai's _combine_llm_outputs
            # (it does `token_usage[k] += v` and Mistral now returns a dict-valued usage
            # field, so summing two dicts raises TypeError). One completion sidesteps the
            # multi-output combine entirely; the score is slightly noisier but valid.
            ResponseRelevancy(strictness=1),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=emb,
        run_config=RunConfig(max_workers=max_workers, timeout=180),
        show_progress=True,
    )
    # result behaves like a dict of column -> aggregate; normalize to plain floats.
    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    out = {k: round(float(scores.get(k, 0.0)), 4) for k in METRIC_KEYS}
    out["n_scored"] = len(answered)
    return out
