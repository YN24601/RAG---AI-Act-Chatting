"""FastAPI serving layer (Day 6-7).

Wraps the Day 5 LangGraph pipeline (`generation.graph.answer_question`) and the
retriever into a deployable HTTP service. Run with the project's no-install
sys.path convention:

    PYTHONPATH=src uvicorn api.app:app
"""
