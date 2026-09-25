import logging
import time
import uuid
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.config import settings
from app.observability import StructuredLoggingCallbackHandler, get_llm_callbacks


def test_get_llm_callbacks_returns_only_logging_handler_when_langfuse_unset(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")

    callbacks = get_llm_callbacks()

    assert len(callbacks) == 1
    assert isinstance(callbacks[0], StructuredLoggingCallbackHandler)


def test_get_llm_callbacks_includes_langfuse_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-test")

    with (
        patch("langfuse.Langfuse") as mock_langfuse,
        patch("langfuse.langchain.CallbackHandler") as mock_handler_cls,
    ):
        mock_handler_cls.return_value = MagicMock(name="langfuse_handler")
        callbacks = get_llm_callbacks()

    mock_langfuse.assert_called_once_with(public_key="pk-test", secret_key="sk-test", host=settings.langfuse_host)
    assert len(callbacks) == 2
    assert isinstance(callbacks[0], StructuredLoggingCallbackHandler)
    assert callbacks[1] is mock_handler_cls.return_value


def test_on_llm_end_logs_latency_and_token_usage(caplog):
    handler = StructuredLoggingCallbackHandler()
    run_id = uuid.uuid4()

    handler.on_llm_start({}, ["prompt"], run_id=run_id)
    time.sleep(0.01)

    message = AIMessage(
        content="hello",
        response_metadata={"model_name": "openai/gpt-oss-120b"},
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    with caplog.at_level(logging.INFO, logger="app.observability"):
        handler.on_llm_end(result, run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_call_completed")
    assert record.model == "openai/gpt-oss-120b"
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    assert record.total_tokens == 15
    assert record.latency_ms > 0


def test_on_llm_error_logs_latency_and_error(caplog):
    handler = StructuredLoggingCallbackHandler()
    run_id = uuid.uuid4()

    handler.on_llm_start({}, ["prompt"], run_id=run_id)

    with caplog.at_level(logging.WARNING, logger="app.observability"):
        handler.on_llm_error(RuntimeError("boom"), run_id=run_id)

    record = next(r for r in caplog.records if r.message == "llm_call_failed")
    assert record.error == "boom"
    assert record.latency_ms is not None
