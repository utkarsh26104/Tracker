"""Structured logging + LLM call instrumentation.

configure_logging() makes settings.log_level actually do something (it was a
dead config value before this module existed) and switches every log line to
one JSON object per line, so a container platform's log viewer (e.g. Cloud
Logging) can parse fields instead of grepping free text.

get_llm_callbacks() is the integration point for latency/token/error
tracking on every Groq call. A callback handler - not per-call-site
instrumentation - is the only way to reach every call site uniformly: two of
this app's five LLM call sites (app/graph/supervisor.py, app/graph/writer.py)
use .with_structured_output(..., method="json_schema", strict=True) without
include_raw=True, so the value they return to the caller is just the parsed
Pydantic object - the underlying AIMessage (and its usage_metadata) is
discarded before the caller ever sees it. A callback's on_llm_start/
on_llm_end fire on the underlying chat-model step *before* that parsing
happens, so this reaches those two sites' real usage data without touching
either file.

Passing get_llm_callbacks() via config={"callbacks": [...]} once, at a
top-level graph.ainvoke() call, is enough to instrument every node's
internal LLM call too - LangGraph propagates callbacks down to every node
through a contextvar (verified against langgraph._internal._runnable.
RunnableCallable.ainvoke and langchain_core.runnables.config.ensure_config),
so supervisor.py/writer.py need zero changes. The three direct (non-graph-
node) .ainvoke() calls in app/graph/master_graph.py - _draft_report_from_
upload, _build_comparison_matrix, _build_client_strategy - run outside any
graph.ainvoke() call, so they get no ambient propagation and need
config={"callbacks": get_llm_callbacks()} passed explicitly at their own
call site instead.

Langfuse is optional and inactive until LANGFUSE_PUBLIC_KEY/SECRET_KEY are
both set (app/config.py) - same no-op-when-unset pattern already used for
API_KEY in app/api/auth.py.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.config import settings

logger = logging.getLogger(__name__)

# Baseline attributes every LogRecord carries - anything else on a record
# (passed via logger.info(..., extra={...})) is a caller-supplied field we
# want to surface in the JSON output, not swallow.
_STANDARD_LOG_RECORD_ATTRS = frozenset(
    logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None).__dict__.keys()
) | {"message", "asctime"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    # force=True: by the time this runs, uvicorn/other libraries may already
    # have called basicConfig or attached their own handlers - force=True
    # replaces them so every log line (ours and theirs) goes through the
    # same JSON formatter instead of a silent mix of formats.
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)


class StructuredLoggingCallbackHandler(BaseCallbackHandler):
    """Logs latency, token usage, and errors for every LLM call. Only
    on_llm_start needs implementing (not on_chat_model_start) -
    BaseCallbackHandler.on_chat_model_start's default deliberately raises
    NotImplementedError specifically to make LangChain's callback manager
    fall back to on_llm_start, which every BaseChatModel (including
    ChatGroq) still triggers."""

    def __init__(self) -> None:
        self._start_times: dict[uuid.UUID, float] = {}

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: uuid.UUID, **kwargs) -> None:
        self._start_times[run_id] = time.monotonic()

    def on_llm_end(self, response: LLMResult, *, run_id: uuid.UUID, **kwargs) -> None:
        model, usage = self._model_and_usage(response)
        logger.info(
            "llm_call_completed",
            extra={
                "event": "llm_call_completed",
                "latency_ms": self._latency_ms(run_id),
                "model": model,
                **usage,
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs) -> None:
        logger.warning(
            "llm_call_failed",
            extra={
                "event": "llm_call_failed",
                "latency_ms": self._latency_ms(run_id),
                "error": str(error),
            },
        )

    def _latency_ms(self, run_id: uuid.UUID) -> float | None:
        start = self._start_times.pop(run_id, None)
        return None if start is None else round((time.monotonic() - start) * 1000, 1)

    @staticmethod
    def _model_and_usage(response: LLMResult) -> tuple[str | None, dict]:
        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            return None, {"input_tokens": None, "output_tokens": None, "total_tokens": None}

        response_metadata = getattr(message, "response_metadata", {}) or {}
        model = response_metadata.get("model_name") or response_metadata.get("model")

        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            return model, {
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            }

        # Fallback for providers/paths that only populate the older field.
        token_usage = (response.llm_output or {}).get("token_usage", {})
        return model, {
            "input_tokens": token_usage.get("prompt_tokens"),
            "output_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
        }


def get_llm_callbacks() -> list[BaseCallbackHandler]:
    callbacks: list[BaseCallbackHandler] = [StructuredLoggingCallbackHandler()]
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        callbacks.append(LangfuseCallbackHandler())
    else:
        logger.debug("Langfuse not configured (LANGFUSE_PUBLIC_KEY/SECRET_KEY unset) - skipping.")
    return callbacks
