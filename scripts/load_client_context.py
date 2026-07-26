"""Loads a client/competitor roster CSV into the client_context ChromaDB
collection, so the Brain agent can ground reports in the consultancy's own
institutional knowledge (which companies are clients, their known
competitors, prior engagement notes) alongside live web research.

Usage:
    python scripts/load_client_context.py path/to/clients.csv

Required CSV columns: client_name, competitor_name
Optional columns: client_details, competitor_details, notes

Safe to re-run after editing the CSV - rows are upserted, keyed by the
(client_name, competitor_name) pair, so edits update existing entries
instead of duplicating them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory.client_context import load_from_csv  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/load_client_context.py path/to/clients.csv")
        sys.exit(1)

    count = load_from_csv(sys.argv[1])
    print(f"Loaded {count} client/competitor row(s) into the client_context collection")
