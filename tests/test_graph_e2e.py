from unittest.mock import patch

import pytest

from app.graph.build_graph import build_thin_graph
from app.graph.state import new_agent_state
from tests.conftest import make_content_driven_supervisor_llm, make_scripted_writer_llm


@pytest.mark.asyncio
async def test_thin_slice_end_to_end(fake_scout_finding):
    """Supervisor -> Scout -> Writer -> FINISH, no Brain/RAG/HITL/Postgres -
    proves the core graph wiring and conditional routing work."""
    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()),
        patch(
            "app.graph.writer.get_writer_llm",
            return_value=make_scripted_writer_llm("# Acme\nAcme raised prices 5%.", sufficient_data=True),
        ),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        graph = build_thin_graph()
        result = await graph.ainvoke(new_agent_state(company="Acme", job_id="e2e-1"))

    assert result["final_report"] == "# Acme\nAcme raised prices 5%."
    assert result["route_history"] == ["Search", "Write", "FINISH"]
    assert len(result["scouted_data"]) == 1


@pytest.mark.asyncio
async def test_thin_slice_loops_back_to_search_when_writer_flags_insufficient(fake_scout_finding):
    from unittest.mock import AsyncMock, MagicMock

    from app.graph.state import RouteDecision, WriterOutput

    supervisor_calls = iter(
        [
            RouteDecision(next="Search", reasoning="need data"),
            RouteDecision(next="Write", reasoning="try writing"),
            RouteDecision(next="Search", reasoning="writer flagged insufficient data, get more"),
            RouteDecision(next="Write", reasoning="try again"),
            RouteDecision(next="FINISH", reasoning="done"),
        ]
    )
    writer_calls = iter(
        [
            WriterOutput(report_markdown="", sufficient_data=False, missing_info="need pricing data"),
            WriterOutput(report_markdown="# Acme\nComplete.", sufficient_data=True),
        ]
    )

    async def supervisor_ainvoke(_messages):
        return next(supervisor_calls)

    async def writer_ainvoke(_messages):
        return next(writer_calls)

    fake_supervisor_structured = MagicMock()
    fake_supervisor_structured.ainvoke = AsyncMock(side_effect=supervisor_ainvoke)
    fake_supervisor_llm = MagicMock()
    fake_supervisor_llm.with_structured_output.return_value = fake_supervisor_structured

    fake_writer_structured = MagicMock()
    fake_writer_structured.ainvoke = AsyncMock(side_effect=writer_ainvoke)
    fake_writer_llm = MagicMock()
    fake_writer_llm.with_structured_output.return_value = fake_writer_structured

    with (
        patch("app.graph.supervisor.get_supervisor_llm", return_value=fake_supervisor_llm),
        patch("app.graph.writer.get_writer_llm", return_value=fake_writer_llm),
        patch("app.graph.scout.search_company", return_value=[fake_scout_finding]),
    ):
        graph = build_thin_graph()
        result = await graph.ainvoke(new_agent_state(company="Acme", job_id="e2e-2"))

    assert result["route_history"] == ["Search", "Write", "Search", "Write", "FINISH"]
    assert result["final_report"] == "# Acme\nComplete."
