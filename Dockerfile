# Multi-stage image for the EU AI Act QA service (FastAPI + same-origin page).
# Vector store stays in Qdrant Cloud, so the container is stateless — no data/.
# Secrets (MISTRAL_API_KEY / QDRANT_URL / QDRANT_API_KEY) are injected at runtime.

# --- builder: install deps into an isolated prefix for a clean copy ---
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- runtime ---
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/

# PYTHONPATH=src mirrors the script/sys.path convention (no package install).
# HF_HUB_OFFLINE pins the no-network tokenizer behavior already set in code.
# PORT defaults to 7860 (Hugging Face Space's expected app_port).
ENV PYTHONPATH=/app/src \
    HF_HUB_OFFLINE=1 \
    PORT=7860

EXPOSE 7860

# Liveness probe via stdlib urllib (no curl in slim). Honors $PORT.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','7860')+'/health').status==200 else 1)"

# Shell form so ${PORT} expands; HF uses 7860, local can override with -e PORT=.
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}
