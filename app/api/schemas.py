from typing import Literal

from pydantic import BaseModel

from app.graph.state import CompanyJobStatus


class RunRequest(BaseModel):
    companies: list[str]
    search_days: int = 30  # how far back Scout's web search looks - see the UI's recency dropdown
    client_company: str | None = None  # if set, final synthesis becomes a client-focused strategy
    company_urls: dict[str, str] = {}  # optional company -> URL, seeded as a finding before Scout runs


class ApproveRequest(BaseModel):
    job_id: str
    companies: list[str]
    client_company: str | None = None  # re-sent from /run since there's no server-side job record


class RetryRequest(BaseModel):
    job_id: str
    company: str


class RetryResponse(BaseModel):
    company: str
    status: CompanyJobStatus
    report: str | None
    route_history: list[str]
    error: str | None = None  # user-facing failure reason if status is FAILED (see describe_exception)


class RunResponse(BaseModel):
    job_id: str
    company_statuses: dict[str, CompanyJobStatus]
    company_reports: dict[str, str]
    company_route_histories: dict[str, list[str]]
    company_errors: dict[str, str] = {}  # user-facing failure reason, keyed by company
    comparison_matrix: str | None


class HistoryEntry(BaseModel):
    job_id: str
    company: str | None
    report_type: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]


class UploadClientFileResponse(BaseModel):
    company: str
    source_filename: str
    chunk_count: int


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskReportRequest(BaseModel):
    report_context: str  # the report/strategy markdown the question should be grounded in
    question: str
    history: list[ChatTurn] = []


class AskReportResponse(BaseModel):
    answer: str
