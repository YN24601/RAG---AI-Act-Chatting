"""Experiment tracking for the eval run (Day 8-9): MLflow + LangSmith dataset.

MLflow: one experiment `aiact-rag-eval`, a parent run holding the baseline-vs-
structure comparison table, and one nested run per strategy with its params,
metrics and per-question artifacts — so the two configs sit side by side in the UI.

LangSmith: persist the eval set as a versioned dataset so it can drive regression-
able online evals later. Guarded on LANGSMITH_API_KEY (skips cleanly if unset).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from ingestion.schema import PROJECT_ROOT

from .schema import EvalItem

# ---- pure: comparison table (unit-testable, also written straight into README) ----

_RAGAS_LABELS = [
    ("faithfulness", "faithfulness"),
    ("answer_relevancy", "answer_relevancy"),
    ("llm_context_precision_with_reference", "context_precision"),
    ("context_recall", "context_recall"),
]


def build_comparison_table(summaries: List[dict]) -> str:
    """Markdown table comparing strategies across RAGAS + refusal + latency.

    `summaries` is a list of per-strategy dicts (see scripts/evaluate.py) with keys
    `strategy`, `ragas`, `refusal`, `latency`. Pure — no I/O.
    """
    cols = [s["strategy"] for s in summaries]
    header = "| metric | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]

    def row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(fn(s) for s in summaries) + " |"

    for key, label in _RAGAS_LABELS:
        rows.append(row(label, lambda s, k=key: f"{s['ragas'].get(k, 0.0):.3f}"))
    rows.append(row("refusal_accuracy", lambda s: f"{s['refusal']['accuracy']:.3f}"))
    rows.append(row("refusal_recall", lambda s: f"{s['refusal']['recall']:.3f}"))
    rows.append(row("under-refusals (FN)", lambda s: str(s["refusal"]["false_negatives"])))
    rows.append(row("latency p95 (answer, s)", lambda s: f"{s['latency']['answer_branch']['p95']:.2f}"))
    rows.append(row("latency p95 (refuse, s)", lambda s: f"{s['latency']['refuse_branch']['p95']:.2f}"))
    return "\n".join(rows)


def _flat_metrics(summary: dict) -> dict:
    """Flatten a strategy summary into scalar MLflow metrics."""
    m: dict = {}
    for key, _ in _RAGAS_LABELS:
        m[f"ragas_{key}"] = float(summary["ragas"].get(key, 0.0))
    for k in ("accuracy", "precision", "recall", "f1", "false_negatives", "false_positives"):
        m[f"refusal_{k}"] = float(summary["refusal"][k])
    lat = summary["latency"]
    for branch in ("overall", "answer_branch", "refuse_branch"):
        m[f"latency_{branch}_mean"] = float(lat[branch]["mean"])
        m[f"latency_{branch}_p95"] = float(lat[branch]["p95"])
    return m


# ---- artifact writers ----

def _write_results_csv(path: Path, results) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "category", "should_refuse", "refused", "grade", "latency_s", "error", "question", "answer"])
        for r in results:
            w.writerow([
                r.item.id, r.item.category, r.item.should_refuse, r.refused,
                r.grade, r.latency_s, r.error, r.item.question, r.answer,
            ])


# ---- MLflow ----

MLFLOW_DB_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def log_mlflow(summaries: List[dict], experiment: str = "aiact-rag-eval") -> Optional[str]:
    """Log a parent 'compare' run with nested per-strategy runs. Returns parent run id."""
    import mlflow  # local import: dev-only dep, not needed to import this module

    # An explicit MLFLOW_TRACKING_URI (e.g. a remote server) still wins.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=str(PROJECT_ROOT / "mlruns" / experiment))
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="compare") as parent:
        table = build_comparison_table(summaries)
        with tempfile.TemporaryDirectory() as td:
            cmp_path = Path(td) / "comparison.md"
            cmp_path.write_text(table, encoding="utf-8")
            mlflow.log_artifact(str(cmp_path))

            for s in summaries:
                with mlflow.start_run(run_name=s["strategy"], nested=True):
                    mlflow.log_params(s["params"])
                    mlflow.log_metrics(_flat_metrics(s))
                    csv_path = Path(td) / f"results_{s['strategy']}.csv"
                    json_path = Path(td) / f"results_{s['strategy']}.json"
                    _write_results_csv(csv_path, s["results"])
                    json_path.write_text(json.dumps(
                        {"strategy": s["strategy"], "ragas": s["ragas"],
                         "refusal": s["refusal"], "latency": s["latency"]},
                        indent=2), encoding="utf-8")
                    mlflow.log_artifact(str(csv_path))
                    mlflow.log_artifact(str(json_path))
        return parent.info.run_id


# ---- LangSmith dataset ----

def push_langsmith_dataset(items: List[EvalItem], name: str = "aiact-eval-v1") -> Optional[str]:
    """Persist the eval set as a LangSmith dataset (idempotent). Skips if no API key.

    Returns the dataset name on create, None if skipped (no key) or already exists.
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("  LangSmith: LANGSMITH_API_KEY not set — skipping dataset upload.")
        return None
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=name):
        print(f"  LangSmith: dataset {name!r} already exists — skipping (bump the version to re-create).")
        return None
    ds = client.create_dataset(
        dataset_name=name,
        description="EU AI Act RAG eval set (Day 8-9): Q&A with ground truth + refusal traps.",
    )
    client.create_examples(
        dataset_id=ds.id,
        inputs=[{"question": it.question, "strategy": "structure"} for it in items],
        outputs=[{
            "ground_truth": it.ground_truth,
            "reference_units": it.reference_units,
            "should_refuse": it.should_refuse,
            "category": it.category,
        } for it in items],
    )
    print(f"  LangSmith: created dataset {name!r} with {len(items)} examples.")
    return name
