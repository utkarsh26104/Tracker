from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class RouteDecision(BaseModel):
    """Supervisor's strict structured output."""

    next: Literal["Search", "Analyze", "Write", "FINISH"]
    reasoning: str
    instructions: str | None = None


class ScoutFinding(BaseModel):
    source_url: str
    title: str
    published_date: str | None = None
    snippet: str
    fetched_at: datetime


class HistoricalMatch(BaseModel):
    text: str
    similarity_score: float
    source_date: str | None = None
    sentiment: Literal["positive", "neutral", "negative"] | None = None


class ClientContextMatch(BaseModel):
    """A row from the consultancy's own client/competitor roster (CSV-loaded
    via scripts/load_client_context.py), matched because the researched
    company appears as either the client or the named competitor."""

    client_name: str
    client_details: str | None = None
    competitor_name: str
    competitor_details: str | None = None
    notes: str | None = None


class WriterOutput(BaseModel):
    """Structured writer output - lets the LLM self-report sufficiency."""

    report_markdown: str
    sufficient_data: bool
    missing_info: str | None = None


class AgentState(TypedDict):
    """LangGraph channel state for a single company's cyclic graph run."""

    company: str
    job_id: str
    search_days: int
    messages: Annotated[list[AnyMessage], add_messages]
    scouted_data: list[ScoutFinding]
    historical_context: list[HistoricalMatch]
    client_context: list[ClientContextMatch]
    sentiment_summary: str | None
    final_report: str | None
    route_history: list[str]
    pending_instructions: str | None
    loop_count: int
    max_loops: int
    insufficient_data_flag: bool
    error: str | None


class CompanyJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    DONE = "done"
    FAILED = "failed"


class MasterComparisonState(BaseModel):
    job_id: str
    companies: list[str]
    company_statuses: dict[str, CompanyJobStatus] = {}
    company_reports: dict[str, str] = {}
    company_route_histories: dict[str, list[str]] = {}
    comparison_matrix: str | None = None
    created_at: datetime
    completed_at: datetime | None = None


def new_agent_state(company: str, job_id: str, max_loops: int = 6, search_days: int = 30) -> AgentState:
    return AgentState(
        company=company,
        job_id=job_id,
        search_days=search_days,
        messages=[],
        scouted_data=[],
        historical_context=[],
        client_context=[],
        sentiment_summary=None,
        final_report=None,
        route_history=[],
        pending_instructions=None,
        loop_count=0,
        max_loops=max_loops,
        insufficient_data_flag=False,
        error=None,
    )
