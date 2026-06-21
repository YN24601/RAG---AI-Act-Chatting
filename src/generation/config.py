"""Settings for the generation / orchestration layer (Day 5).

Secrets and .env loading are reused from retrieval.config (single source of
truth); this module only holds generation- and grade-specific parameters.
"""
from __future__ import annotations

# --- Generation model (Mistral chat) ---
GEN_MODEL = "mistral-small-latest"
GEN_TEMPERATURE = 0.0  # grounded legal QA: deterministic, no creative drift

# --- Grade (retrieve -> grade -> generate) ---
# Score gate: below this top-hit score we refuse outright without spending an
# LLM call. Threshold sits between the measured in-scope floor (~0.72, e.g. the
# "deployer definition" query) and out-of-scope ceiling (~0.62, e.g. unrelated
# questions) from the Day 3-4 retrieval probe.
GRADE_MIN_SCORE = 0.65
GRADE_USE_LLM = True  # after the score gate, also ask the LLM if context is relevant

# --- Generation context ---
ANSWER_TOP_N = 5  # how many retrieved chunks to RETRIEVE as answer candidates

# Per-hit selection before generation. The score gate only vets hits[0]; without
# this, all ANSWER_TOP_N hits (incl. weak tail ones, e.g. 0.55 under a 0.85 top)
# get fed to the prompt and dilute the grounding. We keep the top hit plus only
# those close to it, using two complementary, query-robust criteria:
#   - ANSWER_MIN_SCORE: absolute floor; a hit below this is noise even if recalled.
#   - ANSWER_REL_DROP:  relative floor; drop hits that fall this far below the top
#     hit (a large gap signals a topic shift, and cosine scores drift per query so
#     a relative band travels better than a second hard threshold).
ANSWER_MIN_SCORE = 0.55
ANSWER_REL_DROP = 0.10

# --- Refusal (deterministic; never hand off to the LLM to phrase) ---
REFUSAL_TEXT = (
    "I cannot confirm this from the available EU AI Act provisions. "
    "The retrieved text does not contain a sufficient basis to answer. "
    "For a definitive interpretation, please consult the official text or a qualified professional."
)

# Sentinel the answer model emits (instead of paraphrasing REFUSAL_TEXT) when the
# context can't support an answer. graph.finalize_answer maps it to the canonical
# REFUSAL_TEXT and sets refused=True, so an in-generation refusal is verbatim and
# correctly flagged — not silently mislabeled as a successful answer.
INSUFFICIENT_SENTINEL = "INSUFFICIENT_CONTEXT"

# Tracing is governed entirely by the LANGSMITH_* env vars in .env
# (LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT); no constant here,
# so there's no dead config that silently disagrees with the environment.
