# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch, installed BEFORE the main deps so the `torch>=2.5`
# constraint in pyproject.toml is already satisfied and pip never resolves
# PyPI's default CUDA-bundled build (multi-GB of nvidia-cublas-cu12 etc.) -
# dead weight on Cloud Run, which is CPU-only.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY app ./app
COPY scripts ./scripts
# Main deps only - the "dev" extra (pytest/pytest-asyncio) is deliberately
# not installed in the runtime image.
RUN pip install --no-cache-dir -e .

# Bake both FinBERT models into the image (see app/config.py's defaults) so
# a cold start never hits huggingface.co or pays billed request time
# downloading ~400MB of weights - app/memory/embeddings.py and
# app/memory/sentiment.py load these same two models by name at runtime, so
# this populates a real cache hit, not a partial one. HF_HOME is pinned to
# an explicit path (not the default ~/.cache/huggingface) because this bake
# runs as root but the app runs as appuser below - two different users'
# home directories would otherwise resolve to two different cache paths,
# silently defeating the whole point of baking the models in.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from transformers import AutoTokenizer, AutoModel, pipeline; \
AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone'); \
AutoModel.from_pretrained('yiyanghkust/finbert-tone'); \
pipeline('sentiment-analysis', model='ProsusAI/finbert')"

# chown, not just useradd - everything under /app up to this point (app
# code, the HF cache above, and the not-yet-created ./data/chroma that
# ChromaDB creates on first use) is owned by root. Without this, ChromaDB's
# Rust bindings fail hard at startup with "Permission denied (os error 13)"
# the moment they try to create their persistence directory as appuser.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV APP_ENV=production
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
