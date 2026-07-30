import time
from contextlib import asynccontextmanager
from weakref import WeakKeyDictionary

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


_CONNECTION_CHECK_INTERVAL_SECONDS = 30.0
_last_checked: "WeakKeyDictionary" = WeakKeyDictionary()


async def _check_connection_if_stale(conn) -> None:
    """psycopg_pool's plain AsyncConnectionPool.check_connection runs
    unconditionally on *every* checkout, not just occasionally - a real
    network round-trip to Neon each time, which is meaningful cost during a
    single company's research loop (many checkpoint writes checking out a
    connection in quick succession). Neon's free tier drops connections
    specifically for sitting *idle*, not from being reused quickly and
    repeatedly, so skip the round-trip if this exact connection object was
    already confirmed alive recently - only a connection that's gone
    unused for a while (the actual staleness risk) pays the check. Keyed
    by weak reference so entries for closed/replaced connections don't
    leak."""
    now = time.monotonic()
    last = _last_checked.get(conn)
    if last is not None and now - last < _CONNECTION_CHECK_INTERVAL_SECONDS:
        return
    await AsyncConnectionPool.check_connection(conn)
    _last_checked[conn] = now


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
        # Neon's free tier auto-suspends its compute after a period of
        # inactivity and silently drops idle connections - without this, a
        # dead connection can still look fine to the pool and get handed
        # out anyway, only failing (or hanging on the underlying dead TCP
        # socket, observed taking minutes) once a real query is issued on
        # it. See _check_connection_if_stale's docstring for why this isn't
        # just the raw AsyncConnectionPool.check_connection.
        check=_check_connection_if_stale,
    )
    await pool.open(wait=True)
    try:
        await make_checkpointer(pool).setup()
        yield pool
    finally:
        await pool.close()
