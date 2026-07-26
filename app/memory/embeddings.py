import threading

import torch
from transformers import AutoModel, AutoTokenizer

from app.config import settings

# See sentiment.py's _get_sentiment_pipeline for why this needs a real lock
# instead of just @lru_cache - concurrent first-callers (e.g. two companies'
# brain_node running at once) can otherwise race into model loading together.
_tokenizer_and_model = None
_load_lock = threading.Lock()


def _get_tokenizer_and_model():
    global _tokenizer_and_model
    if _tokenizer_and_model is None:
        with _load_lock:
            if _tokenizer_and_model is None:
                tokenizer = AutoTokenizer.from_pretrained(settings.finbert_embed_model)
                model = AutoModel.from_pretrained(settings.finbert_embed_model)
                model.eval()
                _tokenizer_and_model = (tokenizer, model)
    return _tokenizer_and_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Mean-pooled, attention-mask-aware, L2-normalized FinancialBERT embeddings."""
    if not texts:
        return []

    tokenizer, model = _get_tokenizer_and_model()
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()

    summed = torch.sum(token_embeddings * attention_mask, dim=1)
    counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / counts

    normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
    return normalized.tolist()


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
