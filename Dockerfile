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
# a Cloud Run cold start never hits huggingface.co or pays billed request
# time downloading ~400MB of weights - app/memory/embeddings.py and
# app/memory/sentiment.py load these same two models by name at runtime, so
# this populates a real cache hit, not a partial one.
RUN python -c "\
from transformers import AutoTokenizer, AutoModel, pipeline; \
AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone'); \
AutoModel.from_pretrained('yiyanghkust/finbert-tone'); \
pipeline('sentiment-analysis', model='ProsusAI/finbert')"

RUN useradd --create-home --uid 1000 appuser
USER appuser

ENV APP_ENV=production
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
