from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.report_chat import _MAX_CONTEXT_CHARS, _MAX_HISTORY_TURNS, answer_report_question


def _mock_llm(captured: list):
    llm = MagicMock()

    async def _ainvoke(messages):
        captured.append(messages)
        return SimpleNamespace(content="Mock answer")

    llm.ainvoke = AsyncMock(side_effect=_ainvoke)
    return llm


@pytest.mark.asyncio
async def test_answer_is_grounded_in_report_context():
    captured = []
    with patch("app.services.report_chat.get_writer_llm", return_value=_mock_llm(captured)):
        answer = await answer_report_question(
            report_context="Acme's pricing rose 10% due to competitor pressure.",
            question="Why did pricing rise?",
            history=[],
        )

    assert answer == "Mock answer"
    system_prompt = captured[0][0][1]
    assert "Acme's pricing rose 10%" in system_prompt
    human_message = captured[0][-1]
    assert human_message == ("human", "Why did pricing rise?")


@pytest.mark.asyncio
async def test_conversation_history_is_included_and_capped():
    captured = []
    long_history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(20)]

    with patch("app.services.report_chat.get_writer_llm", return_value=_mock_llm(captured)):
        await answer_report_question(report_context="Some report.", question="Follow-up?", history=long_history)

    messages = captured[0]
    # system prompt + capped history + the new question
    assert len(messages) == 1 + _MAX_HISTORY_TURNS + 1
    # only the most recent turns survive, oldest ones are dropped
    included_contents = [content for _, content in messages[1:-1]]
    assert included_contents == [t["content"] for t in long_history[-_MAX_HISTORY_TURNS:]]


@pytest.mark.asyncio
async def test_long_report_context_gets_truncated():
    captured = []
    huge_context = "x" * (_MAX_CONTEXT_CHARS + 5000)

    with patch("app.services.report_chat.get_writer_llm", return_value=_mock_llm(captured)):
        await answer_report_question(report_context=huge_context, question="Anything?", history=[])

    system_prompt = captured[0][0][1]
    assert len(system_prompt) < len(huge_context)
    assert "..." in system_prompt
