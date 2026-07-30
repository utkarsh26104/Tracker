"""Requires a reachable Postgres via .env's DATABASE_URL - skipped otherwise
(see the pg_pool fixture in conftest.py)."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from psycopg_pool import AsyncConnectionPool

from app.db.checkpointer import make_checkpointer
from app.graph.build_graph import build_graph
from app.graph.state import RouteDecision, new_agent_state
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


def test_pool_has_a_connection_health_check_configured(pg_pool):
    """Regression guard for a real hang observed in production use: Neon's
    free tier silently drops idle connections after its compute
    auto-suspends, and without an active liveness check, the pool can hand
    out a dead connection that then hangs (rather than failing fast) the
    first time something tries to actually use it - surfacing as a
    many-minutes-long, mysterious request timeout with no clear cause."""
    assert pg_pool._check is AsyncConnectionPool.check_connection


@pytest.mark.asyncio
async def test_crash_mid_run_resumes_from_checkpoint_instead_of_restarting(pg_pool, fake_scout_finding):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Phase 1: crash after the first Search decision, before the graph finishes.
    calls = iter([RouteDecision(next="Search", reasoning="need data"), RuntimeError("simulated crash")])

    async def _invoke(_messages):
        result = next(calls)
        if isinstance(result, Exception):
            raise result
        return result

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_invoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured

    graph = build_graph(checkpointer=make_checkpointer(pg_pool))

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=fake_llm),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        with pytest.raises(RuntimeError):
            await graph.ainvoke(new_agent_state(company="Acme", job_id=thread_id), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.values["route_history"] == ["Search"]
    assert len(snapshot.values["scouted_data"]) == 1

    # Phase 2: "restart" with a fresh checkpointer/graph, same thread_id - the
    # already-completed Search step must not re-run.
    graph2 = build_graph(checkpointer=make_checkpointer(pg_pool))
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch(
            "app.graph.writer.get_writer_llm",
            return_value=make_scripted_writer_llm("# Acme\nDone.", True),
        ),
        patch("app.graph.scout.search_company") as mock_search,
    ):
        result = await graph2.ainvoke(None, config=config)
        mock_search.assert_not_called()

    assert result["final_report"] == "# Acme\nDone."
