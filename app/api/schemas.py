from pydantic import BaseModel

from app.graph.state import CompanyJobStatus


class RunRequest(BaseModel):
    companies: list[str]
    search_days: int = 30  # how far back Scout's web search looks - see the UI's recency dropdown
    client_company: str | None = None  # if set, final synthesis becomes a client-focused strategy


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


class RunResponse(BaseModel):
    job_id: str
    company_statuses: dict[str, CompanyJobStatus]
    company_reports: dict[str, str]
    company_route_histories: dict[str, list[str]]
    comparison_matrix: str | None


class HistoryEntry(BaseModel):
    job_id: str
    company: str | None
    report_type: str
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
