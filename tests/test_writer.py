from unittest.mock import patch

import pytest

from app.graph.state import new_agent_state
from app.graph.writer import writer_node
from tests.conftest import make_scripted_writer_llm


@pytest.mark.asyncio
async def test_writer_sets_final_report_when_sufficient():
    state = new_agent_state(company="Acme", job_id="t1")

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=make_scripted_writer_llm("# Acme\nGrowing steadily.", sufficient_data=True),
    ):
        result = await writer_node(state)

    assert result["final_report"] == "# Acme\nGrowing steadily."
    assert result["insufficient_data_flag"] is False


@pytest.mark.asyncio
async def test_writer_flags_insufficient_data_instead_of_fabricating():
    state = new_agent_state(company="Acme", job_id="t1")

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=make_scripted_writer_llm(sufficient_data=False),
    ):
        result = await writer_node(state)

    assert "final_report" not in result
    assert result["insufficient_data_flag"] is True
