import asyncio
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from app.api.auth import require_api_key
from app.api.schemas import (
    ApproveRequest,
    AskReportRequest,
    AskReportResponse,
    HistoryResponse,
    RetryRequest,
    RetryResponse,
    RunRequest,
    RunResponse,
    UploadClientFileResponse,
)
from app.db.history import list_history
from app.graph.master_graph import resume_and_finalize, retry_company, run_map_phase
from app.graph.state import CompanyJobStatus
from app.memory.client_uploads import upsert_client_upload
from app.services.document_parsing import extract_text
from app.services.report_chat import answer_report_question

router = APIRouter(prefix="/tracker", tags=["tracker"], dependencies=[Depends(require_api_key)])


@router.post("/run", response_model=RunResponse)
async def run_tracker(request: RunRequest, http_request: Request) -> RunResponse:
    """Map phase: runs each company's research graph. Each one pauses at the
    HITL gate once a draft report is ready - nothing is published yet. Call
    POST /tracker/approve with the returned job_id to continue."""
    job_id = str(uuid.uuid4())
    pool = http_request.app.state.db_pool

    # The client itself needs its own report for strategy synthesis to have
    # anything to compare competitors against - include it even if the
    # caller only listed competitors.
    companies = list(request.companies)
    if request.client_company and request.client_company not in companies:
        companies.append(request.client_company)

    master_state = await run_map_phase(
        job_id=job_id,
        companies=companies,
        pool=pool,
        search_days=request.search_days,
        client_company=request.client_company,
    )

    return RunResponse(
        job_id=master_state.job_id,
        company_statuses=master_state.company_statuses,
        company_reports=master_state.company_reports,
        company_route_histories=master_state.company_route_histories,
        comparison_matrix=master_state.comparison_matrix,
    )


@router.post("/upload-client-file", response_model=UploadClientFileResponse)
async def upload_client_file(company: str = Form(...), file: UploadFile = File(...)) -> UploadClientFileResponse:
    """Ingest a consultancy-supplied dossier (PDF/.txt/.md) on a client into
    the client_uploads RAG collection. Replaces any prior upload for the
    same company. A future /tracker/run naming this company as client_company
    reuses it (see run_map_phase) instead of always doing a fresh web
    search, as long as it's still recent enough."""
    content = await file.read()
    try:
        text = await asyncio.to_thread(extract_text, file.filename, content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    chunk_count = await asyncio.to_thread(upsert_client_upload, company, file.filename, text)
    return UploadClientFileResponse(company=company, source_filename=file.filename, chunk_count=chunk_count)


@router.post("/ask", response_model=AskReportResponse)
async def ask_about_report(request: AskReportRequest) -> AskReportResponse:
    """Q&A over a report/strategy the caller already has (report_context is
    sent by the client, not looked up server-side) - lets the reviewer ask
    follow-up questions grounded in that exact content, with a bounded
    conversation history for natural follow-ups."""
    answer = await answer_report_question(
        report_context=request.report_context,
        question=request.question,
        history=[turn.model_dump() for turn in request.history],
    )
    return AskReportResponse(answer=answer)


@router.get("/history", response_model=HistoryResponse)
async def get_history(http_request: Request, limit: int = 50) -> HistoryResponse:
    """Every report that has actually been published (approved) so far,
    newest first - individual company reports and comparison matrices alike."""
    pool = http_request.app.state.db_pool
    entries = await list_history(pool, limit=limit)
    return HistoryResponse(entries=entries)


@router.post("/retry", response_model=RetryResponse)
async def retry_tracker(request: RetryRequest, http_request: Request) -> RetryResponse:
    """A reviewer isn't happy with a company's draft report - redirect that
    company's paused graph back into the research loop (Search/Analyze/Write)
    instead of publishing the existing draft, then pause again once a new
    one is ready. Only affects the named company; the rest of the job (and
    any already-approved companies) are untouched."""
    pool = http_request.app.state.db_pool

    try:
        result = await retry_company(pool, job_id=request.job_id, company=request.company)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    report = result.get("final_report")
    return RetryResponse(
        company=request.company,
        status=CompanyJobStatus.AWAITING_APPROVAL if report else CompanyJobStatus.FAILED,
        report=report,
        route_history=result.get("route_history", []),
    )


@router.post("/approve", response_model=RunResponse)
async def approve_tracker(request: ApproveRequest, http_request: Request) -> RunResponse:
    """Reduce phase: resumes each named company's graph past the HITL gate to
    completion, then synthesizes their reports into the Comparison Matrix.
    Companies from the original run that aren't listed here are simply left
    paused (rejected)."""
    pool = http_request.app.state.db_pool

    master_state = await resume_and_finalize(
        job_id=request.job_id, companies=request.companies, pool=pool, client_company=request.client_company
    )

    return RunResponse(
        job_id=master_state.job_id,
        company_statuses=master_state.company_statuses,
        company_reports=master_state.company_reports,
        company_route_histories=master_state.company_route_histories,
        comparison_matrix=master_state.comparison_matrix,
    )
