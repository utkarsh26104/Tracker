import asyncio
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.db.checkpointer import pool_context
from app.db.history import init_history_table
from app.graph.state import RouteDecision, ScoutFinding, WriterOutput


@pytest.fixture(scope="session")
def event_loop_policy():
    # psycopg's async mode needs SelectorEventLoop; pytest-asyncio's default
    # loop on Windows is ProactorEventLoop, which makes the Postgres pool
    # silently fail to connect (times out rather than erroring clearly).
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


def make_content_driven_supervisor_llm():
    """A scripted supervisor LLM whose routing decision is derived from the
    *content* of the state summary it's given, not call order - correct even
    under concurrent, interleaved calls from multiple companies' graphs."""

    async def _ainvoke(messages):
        human_text = messages[1][1]
        if "Final report drafted: True" in human_text:
            return RouteDecision(next="FINISH", reasoning="report is done")
        if "Scouted findings: 0" in human_text:
            return RouteDecision(next="Search", reasoning="need data first")
        return RouteDecision(next="Write", reasoning="enough data scouted")

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


def make_scripted_writer_llm(report_markdown: str = "# Report\nSome findings.", sufficient_data: bool = True):
    async def _ainvoke(_messages):
        return WriterOutput(report_markdown=report_markdown, sufficient_data=sufficient_data)

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


@pytest.fixture
def fake_scout_finding() -> ScoutFinding:
    return ScoutFinding(
        source_url="https://example.com/news",
        title="Company news",
        snippet="Some competitive news snippet about pricing and product launches.",
        fetched_at=datetime.now(timezone.utc),
    )


@pytest.fixture
async def pg_pool():
    """Real Postgres connection pool (Neon or local) via .env's DATABASE_URL.
    Skips the test if no database is reachable, so the rest of the suite
    stays runnable without credentials configured."""
    try:
        async with pool_context() as pool:
            await init_history_table(pool)
            yield pool
    except Exception as e:
        pytest.skip(f"No reachable Postgres for integration tests: {e}")
