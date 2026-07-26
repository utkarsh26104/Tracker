"""A separate ChromaDB collection for the consultancy's own client/competitor
roster (see scripts/load_client_context.py), kept isolated from
vector_store.py's scouted-web-history collection - different kind of content
(curated relationship facts vs. scraped news), and mixing them would let
unrelated web snippets crowd out this deliberately-provided context in
similarity search."""

import csv
import hashlib
import threading

import chromadb

from app.config import settings
from app.graph.state import ClientContextMatch
from app.memory.embeddings import embed_texts

COLLECTION_NAME = "client_context"

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


def _row_id(client_name: str, competitor_name: str) -> str:
    # Stable id from the (client, competitor) pair so re-running the CSV
    # loader updates existing rows instead of duplicating them.
    return hashlib.sha256(f"{client_name}::{competitor_name}".encode()).hexdigest()[:16]


def load_from_csv(path: str) -> int:
    """Expects columns: client_name, client_details, competitor_name,
    competitor_details, notes (header row required; the two *_details
    columns and notes may be blank per-row). Idempotent - safe to re-run
    after editing the CSV, existing (client, competitor) rows get updated
    rather than duplicated."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"client_name", "competitor_name"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")
        rows = [row for row in reader if row.get("client_name") and row.get("competitor_name")]

    return upsert_client_context(rows)


def upsert_client_context(rows: list[dict]) -> int:
    """Each row: client_name, client_details, competitor_name,
    competitor_details, notes (all but the two *_name fields optional)."""
    if not rows:
        return 0

    texts = []
    metadatas = []
    ids = []
    for row in rows:
        client_name = row["client_name"].strip()
        competitor_name = row["competitor_name"].strip()
        client_details = (row.get("client_details") or "").strip()
        competitor_details = (row.get("competitor_details") or "").strip()
        notes = (row.get("notes") or "").strip()

        text = (
            f"Client: {client_name}"
            + (f" ({client_details})" if client_details else "")
            + f". Known competitor: {competitor_name}"
            + (f" ({competitor_details})" if competitor_details else "")
            + (f". Notes: {notes}" if notes else "")
        )
        texts.append(text)
        metadatas.append(
            {
                "client_name": client_name,
                "client_details": client_details,
                "competitor_name": competitor_name,
                "competitor_details": competitor_details,
                "notes": notes,
            }
        )
        ids.append(_row_id(client_name, competitor_name))

    embeddings = embed_texts(texts)
    _get_collection().upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(rows)


def query_client_context(company: str) -> list[ClientContextMatch]:
    """Finds rows where `company` appears as either the client or the named
    competitor - a researched company could plausibly be either."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    matches: dict[str, ClientContextMatch] = {}
    for field in ("client_name", "competitor_name"):
        result = collection.get(where={field: company})
        for meta in result.get("metadatas") or []:
            match = ClientContextMatch(
                client_name=meta["client_name"],
                client_details=meta.get("client_details") or None,
                competitor_name=meta["competitor_name"],
                competitor_details=meta.get("competitor_details") or None,
                notes=meta.get("notes") or None,
            )
            matches[_row_id(match.client_name, match.competitor_name)] = match

    return list(matches.values())
