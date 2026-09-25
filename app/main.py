import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_tracker import router as tracker_router
from app.config import settings
from app.db.checkpointer import pool_context
from app.db.history import init_history_table
from app.memory.seed_data import seed_if_empty
from app.observability import configure_logging

logger = logging.getLogger(__name__)

# Windows note: run this app via `python scripts/serve.py` or
# `uvicorn app.main:app --loop app.winloop:selector_loop_factory` - plain
# `uvicorn app.main:app` uses ProactorEventLoop on win32, which psycopg's
# async mode (used by the Postgres checkpointer) can't run under. See
# app/winloop.py for why a plain asyncio.set_event_loop_policy() call here
# wouldn't fix it - uvicorn's loop_factory bypasses the policy entirely.


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if settings.app_env != "dev" and not settings.api_key:
        logger.warning(
            "APP_ENV=%s but API_KEY is unset - /tracker/* endpoints are unauthenticated "
            "and reachable by anyone with the URL.",
            settings.app_env,
        )
    async with pool_context() as pool:
        app.state.db_pool = pool
        await init_history_table(pool)
        # Ephemeral-filesystem deploy targets (e.g. Render free tier) wipe
        # ChromaDB on every redeploy/restart - re-seed if it's empty so the
        # Brain agent's retrieval step always has something to find.
        await asyncio.to_thread(seed_if_empty)
        yield


app = FastAPI(
    title="Tracker",
    description="Autonomous multi-agent competitive intelligence tracker",
    lifespan=lifespan,
)
app.include_router(tracker_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
