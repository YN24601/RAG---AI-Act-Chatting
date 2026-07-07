"""Run the RAG pipeline over the eval set and collect predictions (Day 8-9).

Calls the exact production entry point (`generation.graph.answer_question`) per
question — so the eval measures the real retrieve->grade->generate|refuse loop,
not a reimplementation. Collects the answer, the authoritative `refused` flag, the
retrieved contexts (for RAGAS), per-question wall-clock latency, and the grade.

Runs sequentially on purpose: Mistral's free tier is rate-limit sensitive and
RAGAS will later fire its own concurrent judge calls; adding request-side
concurrency here would just trip 429s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List

from generation.graph import answer_question

from .schema import EvalItem


@dataclass
class RunResult:
    """One pipeline run over one eval item (carries the item for downstream scoring)."""

    item: EvalItem
    strategy: str
    answer: str
    refused: bool
    grade: str
    contexts: List[str] = field(default_factory=list)  # retrieved hit texts -> RAGAS
    latency_s: float = 0.0
    error: str = ""  # set if the pipeline raised (e.g. PipelineError); result still recorded


def _default_runner(question: str, strategy: str) -> dict:
    return answer_question(question, strategy=strategy)


def run_over_set(
    items: List[EvalItem],
    strategy: str,
    runner: Callable[[str, str], dict] = _default_runner,
    progress: bool = True,
    pause_s: float = 0.0,
) -> List[RunResult]:
    """Execute `strategy` over every item, timing each call.

    `runner` is injectable so tests can drive this offline without touching the
    network. A per-item exception is captured (not raised) so one flaky call can't
    abort the whole run; the item is recorded with `error` set and treated as
    non-refused/empty downstream.

    `pause_s` sleeps between items to smooth request rate — Mistral's free tier can
    return transient "model unavailable" bursts when hit back-to-back (each item is
    2 chat calls: grade + generate). Default 0 keeps offline tests instant.
    """
    results: List[RunResult] = []
    for i, item in enumerate(items, 1):
        if pause_s and i > 1:
            time.sleep(pause_s)
        t0 = time.perf_counter()
        try:
            state = runner(item.question, strategy)
            latency = time.perf_counter() - t0
            hits = state.get("hits", []) or []
            results.append(
                RunResult(
                    item=item,
                    strategy=strategy,
                    answer=state.get("answer", ""),
                    refused=bool(state.get("refused", False)),
                    grade=state.get("grade", ""),
                    contexts=[h.text for h in hits],
                    latency_s=round(latency, 4),
                )
            )
        except Exception as e:  # noqa: BLE001 — record and continue; don't lose the whole run
            latency = time.perf_counter() - t0
            results.append(
                RunResult(
                    item=item,
                    strategy=strategy,
                    answer="",
                    refused=False,
                    grade="error",
                    contexts=[],
                    latency_s=round(latency, 4),
                    error=f"{type(e).__name__}: {e}",
                )
            )
        if progress:
            r = results[-1]
            flag = "REFUSED" if r.refused else ("ERROR" if r.error else "answered")
            print(f"  [{i}/{len(items)}] {item.id:<10} {flag:<8} {r.latency_s:>6.2f}s  {item.question[:56]}")
    return results
