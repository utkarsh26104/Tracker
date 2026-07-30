"""Pure logic tests for _check_connection_if_stale's caching - no real
Postgres needed, unlike most of this project's other checkpointer tests."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.db.checkpointer import _CONNECTION_CHECK_INTERVAL_SECONDS, _check_connection_if_stale


@pytest.mark.asyncio
async def test_skips_round_trip_when_recently_checked():
    conn = MagicMock()
    with patch("app.db.checkpointer.AsyncConnectionPool.check_connection", new=AsyncMock()) as mock_check:
        await _check_connection_if_stale(conn)
        await _check_connection_if_stale(conn)

    mock_check.assert_called_once_with(conn)


@pytest.mark.asyncio
async def test_rechecks_once_the_interval_has_elapsed():
    conn = MagicMock()
    with patch("app.db.checkpointer.AsyncConnectionPool.check_connection", new=AsyncMock()) as mock_check:
        await _check_connection_if_stale(conn)
        later = time.monotonic() + _CONNECTION_CHECK_INTERVAL_SECONDS + 1
        with patch("app.db.checkpointer.time.monotonic", return_value=later):
            await _check_connection_if_stale(conn)

    assert mock_check.call_count == 2


@pytest.mark.asyncio
async def test_tracks_staleness_independently_per_connection():
    conn_a, conn_b = MagicMock(), MagicMock()
    with patch("app.db.checkpointer.AsyncConnectionPool.check_connection", new=AsyncMock()) as mock_check:
        await _check_connection_if_stale(conn_a)
        await _check_connection_if_stale(conn_b)

    # A fresh connection object hasn't been checked before, regardless of
    # whether some *other* connection was just verified.
    assert mock_check.call_count == 2
