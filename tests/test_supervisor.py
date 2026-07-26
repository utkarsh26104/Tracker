from unittest.mock import patch

import pytest

from app.graph.state import new_agent_state
from app.graph.supervisor import route_from_supervisor, supervisor_node
from tests.conftest import make_content_driven_supervisor_llm


@pytest.mark.asyncio
async def test_loop_guardrail_forces_write_before_cap():
    """At the loop cap with no report yet, the guardrail must force a Write
    decision deterministically - without even calling the LLM - so a flaky
    routing model can't loop forever."""
    state = new_agent_state(company="Acme", job_id="t1", max_loops=3)
    state["loop_count"] = 3  # next call would be loop 4, over the cap

    with patch("app.graph.supervisor.get_supervisor_llm") as mock_get_llm:
        result = await supervisor_node(state)

    mock_get_llm.assert_not_called()
    assert result["route_history"][-1].startswith("Write")


@pytest.mark.asyncio
async def test_loop_guardrail_forces_finish_once_report_exists():
    state = new_agent_state(company="Acme", job_id="t1", max_loops=3)
    state["loop_count"] = 3
    state["final_report"] = "# Acme\nDone."

    with patch("app.graph.supervisor.get_supervisor_llm") as mock_get_llm:
        result = await supervisor_node(state)

    mock_get_llm.assert_not_called()
    assert result["route_history"][-1].startswith("FINISH")


@pytest.mark.asyncio
async def test_supervisor_routes_via_structured_llm_output():
    state = new_agent_state(company="Acme", job_id="t1")

    with patch("app.graph.supervisor.get_supervisor_llm", return_value=make_content_driven_supervisor_llm()):
        result = await supervisor_node(state)

    assert result["route_history"] == ["Search"]  # no scouted_data yet -> Search


@pytest.mark.parametrize(
    "decision_text,expected_node",
    [
        ("Search", "scout"),
        ("Analyze", "brain"),
        ("Write", "writer"),
        ("FINISH", "__end__"),
        ("Write (forced: loop cap reached)", "writer"),
    ],
)
def test_route_from_supervisor_mapping(decision_text, expected_node):
    state = new_agent_state(company="Acme", job_id="t1")
    state["route_history"] = [decision_text]
    assert route_from_supervisor(state) == expected_node
