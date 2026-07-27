from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.state import WriterOutput, new_agent_state
from app.graph.writer import FORCED_FINAL_ATTEMPT_INSTRUCTION, writer_node
from tests.conftest import make_scripted_writer_llm


def _capturing_writer_llm(report_markdown: str, sufficient_data: bool, captured: list):
    async def _ainvoke(messages):
        captured.append(messages)
        return WriterOutput(report_markdown=report_markdown, sufficient_data=sufficient_data)

    fake_structured = MagicMock()
    fake_structured.ainvoke = AsyncMock(side_effect=_ainvoke)
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return fake_llm


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


@pytest.mark.asyncio
async def test_forced_instruction_included_when_seeded_company_hits_loop_cap():
    state = new_agent_state(company="LocalBrand", job_id="t1", max_loops=3)
    state["loop_count"] = 4  # exceeds max_loops - this is the guardrail-forced Write
    state["seeded_from_url"] = True
    captured = []

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=_capturing_writer_llm("# LocalBrand\nThin but real.", sufficient_data=False, captured=captured),
    ):
        await writer_node(state)

    system_prompt = captured[0][0][1]
    assert FORCED_FINAL_ATTEMPT_INSTRUCTION.strip() in system_prompt


@pytest.mark.asyncio
async def test_forced_instruction_omitted_when_not_seeded_even_at_loop_cap():
    state = new_agent_state(company="Acme", job_id="t1", max_loops=3)
    state["loop_count"] = 4  # exceeds max_loops, but never seeded from a URL
    captured = []

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=_capturing_writer_llm("# Acme", sufficient_data=False, captured=captured),
    ):
        await writer_node(state)

    system_prompt = captured[0][0][1]
    assert FORCED_FINAL_ATTEMPT_INSTRUCTION.strip() not in system_prompt


@pytest.mark.asyncio
async def test_forced_instruction_omitted_when_seeded_but_within_loop_budget():
    state = new_agent_state(company="LocalBrand", job_id="t1", max_loops=6)
    state["loop_count"] = 2  # comfortably within budget
    state["seeded_from_url"] = True
    captured = []

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=_capturing_writer_llm("# LocalBrand", sufficient_data=True, captured=captured),
    ):
        await writer_node(state)

    system_prompt = captured[0][0][1]
    assert FORCED_FINAL_ATTEMPT_INSTRUCTION.strip() not in system_prompt


@pytest.mark.asyncio
async def test_forced_report_overrides_insufficient_data_when_seeded_and_capped():
    state = new_agent_state(company="LocalBrand", job_id="t1", max_loops=3)
    state["loop_count"] = 4
    state["seeded_from_url"] = True

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=make_scripted_writer_llm("# LocalBrand\nThin but real.", sufficient_data=False),
    ):
        result = await writer_node(state)

    # The Writer self-assessed insufficient_data=False, but this is the
    # last chance for a URL-seeded company - the deterministic override
    # should still publish whatever markdown it produced.
    assert result["final_report"] == "# LocalBrand\nThin but real."
    assert result["insufficient_data_flag"] is False


@pytest.mark.asyncio
async def test_forced_report_does_not_fabricate_from_truly_empty_output():
    state = new_agent_state(company="LocalBrand", job_id="t1", max_loops=3)
    state["loop_count"] = 4
    state["seeded_from_url"] = True

    with patch(
        "app.graph.writer.get_writer_llm",
        return_value=make_scripted_writer_llm("   ", sufficient_data=False),
    ):
        result = await writer_node(state)

    # Even in the forced-final-attempt case, don't publish nothing dressed
    # up as a report - if the model truly produced no content, still flag it.
    assert "final_report" not in result
    assert result["insufficient_data_flag"] is True
