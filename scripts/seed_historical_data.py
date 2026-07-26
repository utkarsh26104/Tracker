"""CLI wrapper for app.memory.seed_data.seed_historical_data - populates
ChromaDB with sample historical competitor snippets so the Brain agent's
retrieval step has something to find on a fresh checkout. Safe to re-run -
upserts are idempotent on id."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.seed_data import seed_historical_data  # noqa: E402

if __name__ == "__main__":
    count = seed_historical_data()
    print(f"Seeded {count} historical snippet(s)")
