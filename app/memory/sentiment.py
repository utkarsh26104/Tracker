import threading

from transformers import pipeline

from app.config import settings

# @lru_cache alone isn't enough here: it dedupes calls *after* one has already
# completed, but doesn't stop two concurrent first-callers (e.g. two
# companies' brain_node hitting this simultaneously via asyncio.to_thread,
# each in its own OS thread) from both racing into pipeline() at once - this
# broke model loading with `NotImplementedError: Cannot copy out of meta
# tensor; no data!` under real concurrent load. Double-checked locking so
# only the first caller loads it and everyone else waits for that instance.
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_sentiment_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = pipeline("sentiment-analysis", model=settings.finbert_sentiment_model)
    return _pipeline


def classify_sentiment(texts: list[str]) -> list[str]:
    """Returns 'positive' | 'negative' | 'neutral' per text via a FinBERT sentiment head.

    Deliberately a different model from embeddings.py's FinancialBERT - one is tuned
    for embedding quality, the other for classification.
    """
    if not texts:
        return []

    results = _get_sentiment_pipeline()(texts, truncation=True, max_length=512)
    return [r["label"].lower() for r in results]


def summarize_sentiment(texts: list[str]) -> str | None:
    if not texts:
        return None

    labels = classify_sentiment(texts)
    counts = {label: labels.count(label) for label in set(labels)}
    total = len(labels)
    breakdown = ", ".join(f"{label}: {count}/{total}" for label, count in sorted(counts.items()))
    return f"Sentiment breakdown across {total} sources - {breakdown}"
