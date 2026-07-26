import threading

import chromadb

from app.config import settings
from app.graph.state import HistoricalMatch
from app.memory.embeddings import embed_texts

COLLECTION_NAME = "competitor_history"

# See embeddings.py's _get_tokenizer_and_model for why this needs a real lock
# instead of just @lru_cache.
_collection = None
_load_lock = threading.Lock()


def _get_collection():
    global _collection
    if _collection is None:
        with _load_lock:
            if _collection is None:
                client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
                _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


def collection_is_empty() -> bool:
    return _get_collection().count() == 0


def upsert_snippets(company: str, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
    if not texts:
        return

    embeddings = embed_texts(texts)
    full_metadatas = [{**m, "company": company} for m in metadatas]
    _get_collection().upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=full_metadatas)


def query_similar(company: str, query_text: str, top_k: int = 5) -> list[HistoricalMatch]:
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"company": company},
    )

    matches = []
    documents = results.get("documents") or [[]]
    distances = results.get("distances") or [[]]
    metadatas = results.get("metadatas") or [[]]

    for doc, distance, meta in zip(documents[0], distances[0], metadatas[0]):
        # Chroma's default space is L2 on normalized vectors; convert to a
        # 0-1 "similarity" that's more intuitive to display than raw distance.
        similarity = max(0.0, 1.0 - (distance / 2.0))
        matches.append(
            HistoricalMatch(
                text=doc,
                similarity_score=similarity,
                source_date=meta.get("published_date"),
                sentiment=meta.get("sentiment"),
            )
        )
    return matches
