"""Server entry point for both local dev and production deploys (e.g. Render).

Prefer this over the bare `uvicorn app.main:app` CLI command on Windows -
see app/winloop.py for why. On Linux (Render, etc.) the custom loop factory
is a no-op (SelectorEventLoop is already the default there), so this script
is safe to use as the deploy start command too.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from app.config import settings

if __name__ == "__main__":
    is_dev = settings.app_env == "dev"
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0" if not is_dev else "127.0.0.1",
        port=port,
        reload=is_dev,
        loop="app.winloop:selector_loop_factory",
    )
