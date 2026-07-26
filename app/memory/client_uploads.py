"""A third ChromaDB collection, separate from client_context.py's curated
CSV roster and vector_store.py's scouted-web-history: raw client-dossier
documents the consultancy uploads themselves (e.g. an existing profile of
E-Vitamins), chunked and embedded so Brain can pull relevant background even
when the document is longer than fits in one prompt. Re-uploading for a
company replaces its prior chunks rather than accumulating duplicates."""

import threading
from datetime import datetime, timezone

import chromadb

from app.config import settings
from app.graph.state import ClientUploadContext
from app.memory.embeddings import embed_texts

COLLECTION_NAME = "client_uploads"

# Keeps a single chunk semantically coherent and comfortably within the
# embedding model's 512-token limit (see embeddings.py's truncation).
_CHUNK_MAX_CHARS = 1500

# Writer's context budget is already tight (Groq's 8K TPM) - cap how much of
# a long dossier gets included, same spirit as writer.py's _MAX_SNIPPET_CHARS.
_MAX_RETURNED_CHARS = 6000

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


def _chunk_text(text: str, max_chars: int = _CHUNK_MAX_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(p[i : i + max_chars] for i in range(0, len(p), max_chars))
            continue
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def upsert_client_upload(company: str, source_filename: str, text: str, uploaded_at: datetime | None = None) -> int:
    """Replaces any prior upload for this company. Returns the chunk count."""
    collection = _get_collection()
    collection.delete(where={"company": company})

    chunks = _chunk_text(text)
    if not chunks:
        return 0

    uploaded_at = uploaded_at or datetime.now(timezone.utc)
    ids = [f"{company}:{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "company": company,
            "source_filename": source_filename,
            "uploaded_at": uploaded_at.isoformat(),
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    embeddings = embed_texts(chunks)
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)


def get_client_upload(company: str) -> ClientUploadContext | None:
    """All of this company's uploaded chunks, reassembled in order and
    truncated to a prompt-safe length - not a similarity search, since we
    want the client's own dossier in full (as much as fits), not a
    query-ranked subset of it."""
    collection = _get_collection()
    result = collection.get(where={"company": company})
    metadatas = result.get("metadatas") or []
    documents = result.get("documents") or []
    if not metadatas:
        return None

    ordered = sorted(zip(metadatas, documents), key=lambda pair: pair[0]["chunk_index"])
    text = "\n\n".join(doc for _, doc in ordered)
    if len(text) > _MAX_RETURNED_CHARS:
        text = text[:_MAX_RETURNED_CHARS].rstrip() + "..."

    first_meta = ordered[0][0]
    return ClientUploadContext(
        text=text,
        source_filename=first_meta["source_filename"],
        uploaded_at=datetime.fromisoformat(first_meta["uploaded_at"]),
    )
