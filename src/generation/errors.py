"""Controlled error type for the QA pipeline.

The graph nodes call external services (Qdrant for retrieval, Mistral for grading
and generation). A raw timeout / 429 / 5xx from those clients would otherwise
bubble out of `answer_question` as an internal stack trace — fine in the CLI, but
a leaked 500 under the Day 6-7 FastAPI layer. We wrap those failures in a single
`PipelineError` carrying a caller-safe message and the failing `stage`; the
original exception is chained (`raise ... from e`) so the real trace stays in the
logs without being surfaced to clients. The API layer catches this one type for a
unified error response.

Note: a hard service failure is deliberately NOT turned into a refusal. `refused`
is the authoritative metric for "the corpus could not support an answer"; mapping
an outage to a refusal would corrupt it. An outage is an error, not a refusal.
"""
from __future__ import annotations


class PipelineError(RuntimeError):
    """An external service backing a pipeline stage failed.

    `stage` is one of "retrieve" | "generate" (grading degrades gracefully instead
    of raising — see graph.grade).
    """

    def __init__(self, stage: str, message: str):
        self.stage = stage
        super().__init__(f"[{stage}] {message}")
