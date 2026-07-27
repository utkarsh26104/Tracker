import asyncio
import logging
import re
from datetime import datetime, timezone

import groq
import psycopg
import requests

from app.db.checkpointer import make_checkpointer
from app.db.history import get_recent_company_report, save_report
from app.graph.build_graph import build_graph
from app.graph.state import (
    CLIENT_REPORT_CACHE_MAX_AGE_DAYS,
    ClientUploadContext,
    CompanyJobStatus,
    MasterComparisonState,
    new_agent_state,
)
from app.llm.groq_client import get_writer_llm
from app.memory.client_context import query_client_context
from app.memory.client_uploads import get_client_upload
from app.services.scraping import fetch_company_site_findings

logger = logging.getLogger(__name__)

# A consultancy typically re-runs this tool many times for the same client
# against different competitor sets - re-researching the client itself on
# every single run burns Tavily/Groq quota for data that hasn't gone stale.
# Competitors are never cached this way (only the named client_company),
# since the whole point of a run is fresh intel on them specifically.
# (CLIENT_REPORT_CACHE_MAX_AGE_DAYS itself lives in state.py - see its
# comment there for why.)

COMPARISON_SYSTEM_PROMPT_TEMPLATE = """You are synthesizing several single-company competitive
intelligence reports into one executive "Competitive Landscape Comparison Matrix."
Today's date is {today} - judge recency relative to today, not your own training cutoff.
Produce a concise Markdown report: a comparison table (pricing, recent moves, sentiment)
followed by a short synthesis of how these companies compare to each other."""

STRATEGY_SYSTEM_PROMPT_TEMPLATE = """You are a strategic advisor preparing a briefing for our
client, {client_company}. Today's date is {today} - judge recency relative to today, not your
own training cutoff.

You have {client_company}'s own competitive-intelligence report, reports on each of its known
competitors, and (if present) notes from our firm's own records about this client relationship -
treat those notes as authoritative background, not just another finding.

Produce a concise Markdown "Strategic Recommendation" document with these sections:
1. **Competitive Comparison** - a table comparing {client_company} against each competitor
   (pricing, recent moves, sentiment). {client_company} is the client this whole report is
   written for, not one of its own competitors - label the row-identifying column "Company", not
   "Competitor", and mark {client_company}'s own row clearly (e.g. "{client_company} (our
   client)") so it's never confused for one of the competitor rows.
2. **Sentiment & Market Position** - how {client_company} compares in public/market sentiment
   and positioning against the field.
3. **Financial & Business Analysis** - pricing strategy, product moves, and business positioning
   differences that matter competitively.
4. **Recommended Strategy** - 3-5 concrete, actionable recommendations {client_company} could
   act on in direct response to what its competitors are actually doing. Tie each recommendation
   to a specific finding, not generic business advice - if the data doesn't support a strong
   recommendation in some area, say so rather than filling the gap with platitudes.

Keep it under ~700 words - this has a hard output token budget."""

DOSSIER_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are the Writer agent on a competitive-intelligence
research team. Today's date is {today}. Draft a concise executive Markdown report on {company}
using ONLY the uploaded client dossier below - no web search was performed for this run because
the dossier ({freshness}) is recent enough to stand on its own.

Keep the report under ~500 words. If the dossier doesn't cover something a reader would expect
(e.g. recent pricing moves), say so plainly rather than guessing at it."""

# Transient network/connection blips (Neon connection reset mid-checkpoint-write,
# Tavily connection reset, etc.) observed in real runs. Neither is retried by
# LangGraph's own node-level RetryPolicy: psycopg.OperationalError isn't an
# OSError/ConnectionError subclass at all, and requests.exceptions.ConnectionError
# IS an OSError subclass, which LangGraph's default retry predicate explicitly
# excludes. More fundamentally, checkpoint writes happen in LangGraph's own
# Pregel runtime *after* a node returns, outside any node-level RetryPolicy's
# reach - so retrying belongs at the whole-graph-invocation level instead, which
# is safe because checkpointing means a retry resumes rather than restarts.
_TRANSIENT_EXCEPTIONS = (psycopg.OperationalError, requests.exceptions.ConnectionError, requests.exceptions.Timeout, TimeoutError)

# groq.RateLimitError separately: ChatGroq's own max_retries already retries
# 429s internally, but under real concurrent Map-Reduce load (multiple
# companies sharing one org-wide TPM budget) two companies' internal retries
# can keep colliding on the same recovering budget and both still exhaust
# their retry count. Retrying at the whole-graph level spreads attempts out
# further in time and reuses Groq's own suggested wait instead of guessing.
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)


def _rate_limit_wait_seconds(exc: groq.RateLimitError, attempt: int) -> float:
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        return float(match.group(1)) + 1.0  # small buffer past Groq's own estimate
    return 5.0 * attempt  # fallback if the message format ever changes


def _thread_id(job_id: str, company: str) -> str:
    return f"{job_id}:{company}"


async def _ainvoke_with_retry(graph, initial_input, config: dict, max_attempts: int = 4):
    current_input = initial_input
    for attempt in range(1, max_attempts + 1):
        try:
            return await graph.ainvoke(current_input, config=config)
        except (*_TRANSIENT_EXCEPTIONS, groq.RateLimitError) as e:
            if attempt >= max_attempts:
                raise
            wait = _rate_limit_wait_seconds(e, attempt) if isinstance(e, groq.RateLimitError) else 2 ** (attempt - 1)
            logger.warning(
                "%s on attempt %d/%d for thread %s (retrying in %.1fs): %s",
                "Rate limit" if isinstance(e, groq.RateLimitError) else "Transient error",
                attempt,
                max_attempts,
                config["configurable"]["thread_id"],
                wait,
                e,
            )
            await asyncio.sleep(wait)
            # Resuming with None input tells LangGraph "continue from the last
            # checkpoint" - but if the failure happened before the very first
            # checkpoint was ever written (e.g. a rate limit on the first
            # Supervisor call), there IS no checkpoint yet, and None input
            # raises langgraph.errors.EmptyInputError instead of retrying.
            # Check what's actually been persisted before deciding.
            snapshot = await graph.aget_state(config)
            current_input = None if snapshot.values else initial_input


async def run_company(
    pool, company: str, job_id: str, search_days: int = 30, seed_url: str | None = None
) -> tuple[str, dict]:
    # Each concurrently-running company gets its own checkpointer instance
    # (own internal lock), sharing the pool for actual connections - see
    # make_checkpointer's docstring for why one shared instance serializes
    # concurrent checkpoint writes even with a pool underneath.
    checkpointer = make_checkpointer(pool)
    graph = build_graph(checkpointer=checkpointer)

    thread_id = _thread_id(job_id, company)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = new_agent_state(company=company, job_id=thread_id, search_days=search_days)

    if seed_url:
        # Small/local businesses often have thin-to-nonexistent Tavily
        # coverage, which otherwise drives the Supervisor through repeated
        # fruitless Search loops (observed: 6-loop cap hit, then the final
        # forced Write still had nothing concrete to work from). Seeding a
        # known-good URL (e.g. the company's own site) up front - plus a
        # light same-domain crawl for product/discount/review pages, see
        # fetch_company_site_findings - gives the Writer something real
        # even if Tavily comes up empty.
        seed_findings = await asyncio.to_thread(fetch_company_site_findings, seed_url)
        if seed_findings:
            initial_state["scouted_data"] = seed_findings
            initial_state["route_history"] = [
                f"Seeded with {len(seed_findings)} page(s) from provided URL: {seed_url}"
            ]

    result = await _ainvoke_with_retry(graph, initial_state, config)
    return company, result


async def run_company_with_cached_report(
    pool, company: str, job_id: str, cached_report: str, route_history_label: str | None = None
) -> tuple[str, dict]:
    """Skip Scout/Brain/Writer entirely and seed the graph with an already-
    known-good report. max_loops=0 makes the Supervisor's deterministic
    guardrail (not the LLM) force FINISH on its very first call, since
    final_report is already set - so this still creates a real checkpointed
    thread that pauses at the normal HITL gate like any other company. That
    means the reviewer sees it in the review step same as a fresh result,
    and the existing "try again" button (retry_company) still works if they
    want a real search instead of the cached one."""
    checkpointer = make_checkpointer(pool)
    graph = build_graph(checkpointer=checkpointer)

    thread_id = _thread_id(job_id, company)
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = new_agent_state(company=company, job_id=thread_id, max_loops=0)
    initial_state["final_report"] = cached_report
    initial_state["route_history"] = [
        route_history_label
        or f"Reused cached report (< {CLIENT_REPORT_CACHE_MAX_AGE_DAYS} days old) - no fresh search performed"
    ]
    result = await _ainvoke_with_retry(graph, initial_state, config)
    return company, result


async def _draft_report_from_upload(company: str, upload: ClientUploadContext) -> str:
    """A single Writer call over just the uploaded dossier - no Scout/Brain
    involved. Used when the upload is fresh enough that a live web search
    isn't needed (see run_map_phase); a stale upload instead flows through
    the normal full pipeline, where Brain includes it as background
    alongside real Scout findings."""
    llm = get_writer_llm()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    age_days = (datetime.now(timezone.utc) - upload.uploaded_at).days
    response = await llm.ainvoke(
        [
            (
                "system",
                DOSSIER_ONLY_SYSTEM_PROMPT_TEMPLATE.format(
                    today=today, company=company, freshness=f"uploaded {age_days} day(s) ago"
                ),
            ),
            ("human", f"# Uploaded Client Dossier ({upload.source_filename})\n\n{upload.text}"),
        ]
    )
    return response.content


async def retry_company(pool, job_id: str, company: str) -> dict:
    """A reviewer isn't happy with a drafted report and wants fresh research
    before approving. The company's graph is paused right before
    publish_report with `next` already pointing there (the Supervisor already
    decided FINISH) - simply resuming would go straight to publishing the
    same draft, not redo any work. update_state(..., as_node="supervisor")
    rewrites the checkpoint as if the Supervisor had just decided "Search"
    instead, so the existing conditional edge routes back into the research
    loop on resume. Prior scouted_data/historical_context are left untouched
    (only the listed keys are updated) so Scout adds to what's already been
    found rather than starting over from nothing.
    """
    checkpointer = make_checkpointer(pool)
    graph = build_graph(checkpointer=checkpointer)

    thread_id = _thread_id(job_id, company)
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await graph.aget_state(config)
    if not snapshot.values:
        raise ValueError(f"No existing research found for {company} in job {job_id}")

    await graph.aupdate_state(
        config,
        {
            "final_report": None,
            "route_history": snapshot.values["route_history"] + ["Search (forced: user requested re-research)"],
            "pending_instructions": (
                "The previous draft wasn't satisfactory to the reviewer - search for additional, "
                "more specific, or more recent information before writing again."
            ),
            "loop_count": 0,  # fresh loop budget for the retry
        },
        as_node="supervisor",
    )

    return await _ainvoke_with_retry(graph, None, config)


async def run_map_phase(
    job_id: str,
    companies: list[str],
    pool,
    search_days: int = 30,
    client_company: str | None = None,
    company_urls: dict[str, str] | None = None,
) -> MasterComparisonState:
    """Map: run each company's graph concurrently in isolation. Each run pauses
    at the HITL gate (interrupt_before=["publish_report"]) once a report is
    drafted - nothing is published or compared yet. Call resume_and_finalize
    once a human has reviewed and approved which companies to include.

    If client_company names one of the companies and we already have a
    report_history entry for it younger than CLIENT_REPORT_CACHE_MAX_AGE_DAYS,
    that company reuses the cached report instead of researching it again -
    see run_company_with_cached_report. Failing that, if the consultancy has
    uploaded a client dossier (app/memory/client_uploads.py) younger than
    the same threshold, a report is drafted from just that dossier instead
    of a live search. A stale or absent dossier falls through to the normal
    full pipeline, where Brain includes it as background regardless of age -
    so a stale upload still gets mixed with fresh Scout findings rather than
    being ignored outright.

    company_urls optionally maps a company to a URL to seed findings from
    before Scout runs (see run_company/fetch_company_site_findings) - for
    companies with thin-to-none Tavily coverage, e.g. small/local
    businesses."""
    company_urls = company_urls or {}
    master_state = MasterComparisonState(
        job_id=job_id,
        companies=companies,
        company_statuses={c: CompanyJobStatus.RUNNING for c in companies},
        created_at=datetime.now(timezone.utc),
    )

    async def _run(company: str) -> tuple[str, dict]:
        if company == client_company:
            cached_report = await get_recent_company_report(pool, company, CLIENT_REPORT_CACHE_MAX_AGE_DAYS)
            if cached_report is not None:
                return await run_company_with_cached_report(pool, company, job_id, cached_report)

            upload = await asyncio.to_thread(get_client_upload, company)
            if upload is not None:
                age_days = (datetime.now(timezone.utc) - upload.uploaded_at).days
                if age_days <= CLIENT_REPORT_CACHE_MAX_AGE_DAYS:
                    drafted_report = await _draft_report_from_upload(company, upload)
                    return await run_company_with_cached_report(
                        pool,
                        company,
                        job_id,
                        drafted_report,
                        route_history_label=(
                            "Reused cached report (drafted from an uploaded dossier, "
                            f"< {CLIENT_REPORT_CACHE_MAX_AGE_DAYS} days old) - no fresh search performed"
                        ),
                    )
        return await run_company(pool, company, job_id, search_days, seed_url=company_urls.get(company))

    results = await asyncio.gather(
        *(_run(company) for company in companies),
        return_exceptions=True,
    )

    for company, outcome in zip(companies, results):
        if isinstance(outcome, BaseException):
            logger.exception("Map phase failed for %s (job %s)", company, job_id, exc_info=outcome)
            master_state.company_statuses[company] = CompanyJobStatus.FAILED
            continue
        _, result = outcome
        master_state.company_route_histories[company] = result.get("route_history", [])
        report = result.get("final_report")
        if report:
            master_state.company_reports[company] = report
            master_state.company_statuses[company] = CompanyJobStatus.AWAITING_APPROVAL
        else:
            master_state.company_statuses[company] = CompanyJobStatus.FAILED

    return master_state


async def resume_one(pool, company: str, job_id: str) -> tuple[str, dict]:
    checkpointer = make_checkpointer(pool)
    graph = build_graph(checkpointer=checkpointer)

    thread_id = _thread_id(job_id, company)
    config = {"configurable": {"thread_id": thread_id}}
    result = await _ainvoke_with_retry(graph, None, config)
    return company, result


async def resume_and_finalize(
    job_id: str, companies: list[str], pool, client_company: str | None = None
) -> MasterComparisonState:
    """Reduce: resume each approved company's graph past the HITL gate to
    completion, then synthesize their reports. If client_company names one of
    the (approved) researched companies, synthesis becomes a client-focused
    Strategic Recommendation (that company vs. the rest, framed as advice for
    it) instead of a neutral Comparison Matrix across all of them."""
    master_state = MasterComparisonState(
        job_id=job_id,
        companies=companies,
        company_statuses={c: CompanyJobStatus.APPROVED for c in companies},
        created_at=datetime.now(timezone.utc),
    )

    results = await asyncio.gather(
        *(resume_one(pool, company, job_id) for company in companies),
        return_exceptions=True,
    )

    for company, outcome in zip(companies, results):
        if isinstance(outcome, BaseException):
            logger.exception("Reduce phase failed for %s (job %s)", company, job_id, exc_info=outcome)
            master_state.company_statuses[company] = CompanyJobStatus.FAILED
            continue
        _, result = outcome
        report = result.get("final_report")
        if report:
            master_state.company_reports[company] = report
            master_state.company_statuses[company] = CompanyJobStatus.DONE
            await save_report(pool, job_id=job_id, company=company, report_type="company", content=report)
        else:
            master_state.company_statuses[company] = CompanyJobStatus.FAILED

    use_strategy_mode = client_company and client_company in master_state.company_reports
    if master_state.company_reports:
        try:
            if use_strategy_mode:
                master_state.comparison_matrix = await _build_client_strategy(
                    client_company, master_state.company_reports
                )
                report_type = "strategy"
            else:
                master_state.comparison_matrix = await _build_comparison_matrix(master_state.company_reports)
                report_type = "comparison_matrix"
            await save_report(
                pool,
                job_id=job_id,
                company=client_company if use_strategy_mode else None,
                report_type=report_type,
                content=master_state.comparison_matrix,
            )
        except Exception:
            # Individual company reports already succeeded and are sitting in
            # master_state - don't let a failure on this last synthesis step
            # (e.g. hitting Groq's daily token cap) throw them away too. The
            # caller can see comparison_matrix is None and retry just this step.
            logger.exception("Comparison/strategy synthesis failed for job %s", job_id)

    master_state.completed_at = datetime.now(timezone.utc)
    return master_state


async def _build_comparison_matrix(company_reports: dict[str, str]) -> str:
    combined = "\n\n".join(f"## {company}\n{report}" for company, report in company_reports.items())
    llm = get_writer_llm()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    response = await llm.ainvoke(
        [
            ("system", COMPARISON_SYSTEM_PROMPT_TEMPLATE.format(today=today)),
            ("human", combined),
        ]
    )
    return response.content


async def _build_client_strategy(client_company: str, company_reports: dict[str, str]) -> str:
    client_report = company_reports[client_company]
    competitor_reports = {c: r for c, r in company_reports.items() if c != client_company}

    parts = [f"# Client: {client_company}\n{client_report}"]
    if competitor_reports:
        parts.append(
            "\n\n".join(f"## Competitor: {c}\n{r}" for c, r in competitor_reports.items())
        )

    # Re-query at synthesis time (not just relying on what the client's own
    # per-company Brain step already saw) so any firm notes on this exact
    # client relationship are explicitly grounding the strategy prompt.
    context_matches = await asyncio.to_thread(query_client_context, client_company)
    if context_matches:
        notes = "\n".join(
            f"- {m.client_name} vs. known competitor {m.competitor_name}" + (f": {m.notes}" if m.notes else "")
            for m in context_matches
        )
        parts.append(f"\n\n## Our firm's records on {client_company}\n{notes}")

    combined = "\n\n".join(parts)
    llm = get_writer_llm()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    response = await llm.ainvoke(
        [
            ("system", STRATEGY_SYSTEM_PROMPT_TEMPLATE.format(client_company=client_company, today=today)),
            ("human", combined),
        ]
    )
    return response.content
