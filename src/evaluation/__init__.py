"""Day 8-9: RAGAS evaluation + MLflow experiment tracking.

Compares the two chunking strategies (baseline vs structure) on a hand-authored
EU AI Act Q&A set. Metrics: the four RAGAS scores on the answered subset plus a
refusal-accuracy metric on the full set (`refused` is the authoritative "no
corpus basis" signal — see README). Runs offline-dev-only; deliberately absent
from requirements.txt so the serving image stays lean.
"""
