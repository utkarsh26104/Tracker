from app.memory.vector_store import collection_is_empty, upsert_snippets

SAMPLE_HISTORY = [
    {
        "company": "Stripe",
        "id": "seed-stripe-1",
        "text": "Stripe announced a flat 2.9% + 30c pricing tier for standard card payments, unchanged from the prior year.",
        "published_date": "2026-01-15",
    },
    {
        "company": "Stripe",
        "id": "seed-stripe-2",
        "text": "Stripe expanded its Radar fraud-detection product with new ML-based risk scoring for enterprise accounts.",
        "published_date": "2026-02-03",
    },
    {
        "company": "Adyen",
        "id": "seed-adyen-1",
        "text": "Adyen reported strong enterprise growth driven by unified commerce adoption across European retailers.",
        "published_date": "2026-01-20",
    },
    {
        "company": "Acme Corp",
        "id": "seed-acme-1",
        "text": "Acme Corp held pricing steady through the prior fiscal year with no major tier changes announced.",
        "published_date": "2025-11-10",
    },
]


def seed_historical_data() -> int:
    """Idempotent - upserts are keyed by id, safe to call repeatedly. Returns
    the number of snippets written."""
    by_company: dict[str, list[dict]] = {}
    for row in SAMPLE_HISTORY:
        by_company.setdefault(row["company"], []).append(row)

    for company, rows in by_company.items():
        upsert_snippets(
            company=company,
            texts=[r["text"] for r in rows],
            metadatas=[{"published_date": r["published_date"]} for r in rows],
            ids=[r["id"] for r in rows],
        )

    return len(SAMPLE_HISTORY)


def seed_if_empty() -> bool:
    """For deploy targets with an ephemeral filesystem (e.g. Render free
    tier): re-seeds ChromaDB on startup if the collection was wiped by a
    redeploy/restart. Returns True if seeding ran."""
    if collection_is_empty():
        seed_historical_data()
        return True
    return False
