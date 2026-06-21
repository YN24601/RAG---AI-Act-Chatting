"""Ask the EU AI Act RAG pipeline end-to-end (Day 5).

retrieve -> grade -> generate | refuse, with LangSmith tracing when configured.

Usage:
    python scripts/ask.py "What AI practices are prohibited?"
    python scripts/ask.py "definition of deployer" --strategy structure --show-context
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation.graph import answer_question  # noqa: E402
from generation.prompts import format_context  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="EU AI Act RAG Q&A (Day 5)")
    ap.add_argument("question", nargs="+", help="the question to ask")
    ap.add_argument("--strategy", choices=["baseline", "structure"], default="structure")
    ap.add_argument("--show-context", action="store_true", help="print the context fed to the LLM")
    args = ap.parse_args()

    question = " ".join(args.question)
    state = answer_question(question, strategy=args.strategy)

    print(f"\nquestion : {question}")
    print(f"strategy : {args.strategy}")
    print(f"grade    : {state['grade']}  ({state.get('grade_reason', '')})")
    print(f"refused  : {state['refused']}")
    print("=" * 78)
    print(f"\n{state['answer']}\n")

    hits = state.get("hits", [])
    # Hits are score-descending; generate() grounds on the top `used_hits` only,
    # so mark the weaker tail that was retrieved but dropped before answering.
    used = state.get("used_hits", len(hits))
    if hits and not state["refused"]:
        print(f"sources ({used}/{len(hits)} used for the answer):")
        for h in hits:
            header = h.metadata.get("context_header") or f"chunk {h.metadata.get('chunk_index')}"
            tag = "" if h.rank <= used else "  (dropped: low score)"
            print(f"  [{h.rank}] score={h.score}  {header}{tag}")
        print()
    if args.show_context:
        print("-" * 78 + "\ncontext fed to LLM:\n")
        print(format_context(hits[:used] if not state["refused"] else hits))
        print()


if __name__ == "__main__":
    main()
