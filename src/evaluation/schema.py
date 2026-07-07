"""Evaluation-set schema + loader (Day 8-9).

One hand-authored, committed Q&A set drives both chunking-strategy runs. Kept in
`data/eval/eval_set.jsonl` (committed, unlike the gitignored data/processed/) so
the ground truth is reproducible and reviewable. Mirrors the pydantic-v2 style of
ingestion.schema.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel, Field

from ingestion.schema import PROJECT_ROOT

# Committed (NOT under data/processed/, which is gitignored). .dockerignore excludes
# all of data/, so the eval set never enters the runtime image — eval is dev-only.
EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_set.jsonl"

# Coarse buckets over the AI Act, plus the two refusal families. Used for
# per-category breakdowns, not for scoring.
EvalCategory = Literal[
    "prohibited",    # Article 5 prohibited practices
    "high_risk",     # Article 6 / Annex III classification
    "gpai",          # Articles 51-55 general-purpose AI models
    "timeline",      # entry-into-force / application dates
    "definition",    # Article 3 definitions (provider, deployer, ...)
    "governance",    # authorities, conformity assessment, penalties
    "out_of_scope",  # off-topic -> must refuse (score gate)
    "trap",          # on-topic but no binding basis in corpus -> must refuse
]


class EvalItem(BaseModel):
    """One evaluation question with its ground truth and expected refusal outcome."""

    id: str
    question: str
    category: EvalCategory
    ground_truth: str  # reference answer; RAGAS reference for context_recall/precision
    reference_units: List[str] = Field(default_factory=list)  # e.g. ["Article 5", "Recital 27"]
    should_refuse: bool = False  # trap/out-of-scope questions the pipeline must refuse


def load_eval_set(path: Path = EVAL_SET_PATH) -> List[EvalItem]:
    """Load + validate the jsonl eval set (one EvalItem per line).

    Raises if the file is missing or any line fails validation (fail loud — a
    silently-truncated eval set would understate coverage).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"eval set not found at {path} — it is committed under data/eval/; "
            f"check you are on a branch that has it."
        )
    items: List[EvalItem] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = EvalItem.model_validate_json(line)
            except Exception as e:  # noqa: BLE001 — annotate which line to fix
                raise ValueError(f"{path}:{lineno} invalid EvalItem: {e}") from e
            if item.id in seen_ids:
                raise ValueError(f"{path}:{lineno} duplicate id {item.id!r}")
            seen_ids.add(item.id)
            items.append(item)
    if not items:
        raise ValueError(f"{path} contained no eval items")
    return items


def eval_set_hash(path: Path = EVAL_SET_PATH) -> str:
    """sha256 of the raw eval-set bytes — logged to MLflow to pin which set a run used.

    Mirrors the content-fingerprint idempotency the index build already uses.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
