from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

# Our custom Pydantic types embedded in AgentState (ScoutFinding, HistoricalMatch)
# need to be explicitly allow-listed for the checkpoint serializer, or a future
# langgraph-checkpoint-postgres version will refuse to deserialize them.
_ALLOWED_MSGPACK_MODULES = [
    ("app.graph.state", "ScoutFinding"),
    ("app.graph.state", "HistoricalMatch"),
]


def make_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """Builds a fresh AsyncPostgresSaver bound to the shared pool.

    AsyncPostgresSaver holds an internal asyncio.Lock that serializes every
    cursor operation for that *instance*, no matter how big the connection
    pool behind it is. Sharing one checkpointer across M4's concurrently
    running per-company graphs serializes all their checkpoint writes on that
    lock - measured ~10s for 3 companies that should take ~2s. Each
    concurrent graph run needs its own checkpointer instance (own lock); they
    can all still share this one pool for actual connections.
    """
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    return AsyncPostgresSaver(conn=pool, serde=serde)


@asynccontextmanager
async def pool_context():
    """Opens the shared connection pool for the app's lifetime and runs
    checkpoint table setup once. Yields the pool, not a checkpointer -
    callers must build a fresh checkpointer per concurrent graph run via
    make_checkpointer(pool)."""
    pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        min_size=1,
        max_size=10,
        open=False,
    )
    await pool.open(wait=True)
    try:
        await make_checkpointer(pool).setup()
        yield pool
    finally:
        await pool.close()
