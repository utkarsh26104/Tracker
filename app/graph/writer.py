from datetime import datetime, timezone

from dateutil import parser as date_parser

from app.graph.state import CLIENT_REPORT_CACHE_MAX_AGE_DAYS, AgentState, ScoutFinding, WriterOutput
from app.llm.groq_client import get_writer_llm

SYSTEM_PROMPT_TEMPLATE = """You are the Writer agent on a competitive-intelligence research team.
Today's date is {today}. Draft a concise executive Markdown report on the target company \
using the scouted web findings and any historical context provided.

Judge recency relative to today's date, not your own training cutoff - a scouted finding or \
historical snippet with an older date is still worth including for context, but say so \
explicitly (e.g. "as of its most recent public earnings, Q4 2024...") rather than presenting \
old information as current. If nothing recent turned up, say that plainly instead of guessing.

If "Client & Competitor Context" is present, it comes from our own firm's client roster, not \
the public web - treat it as authoritative background (e.g. why we're tracking this company, \
who it's known to compete with) and weave it into the framing, but don't just restate it \
verbatim as if it were a new finding.

If "Uploaded Client Dossier" is present, it's a document our firm supplied directly (not the \
public web either) - treat it as a trusted primary source. If it's marked current, you can lean \
on it heavily even where scouted findings are thin. If it's marked stale, still use it for \
background/history, but let the scouted findings (if any) take precedence for anything recent, \
and note where the dossier may be out of date.

If the public web findings are thin (common for smaller or local businesses with little press \
coverage), don't force a report shaped like one for a large public company. Say plainly what \
little was found, and note that primary research (site visits, direct outreach) would be \
needed for a fuller picture - that's still a useful, honest deliverable for a consultancy.

Keep the report under ~500 words. Favor a few well-chosen sections (overview, recent \
moves, sentiment, outlook) over exhaustive detail - this output has a hard token budget, \
and a report that runs long risks being cut off mid-generation and failing to parse.

If the available data is too thin to write a credible report, set sufficient_data=false \
and explain what's missing in missing_info instead of fabricating content."""


# Real web snippets can run to thousands of tokens each; free-tier LLM APIs
# (e.g. Groq's 8K TPM limit) reject a single request that large. Cap per-item
# length and item count so the context stays comfortably within budget
# regardless of how verbose Tavily's results are.
_MAX_SNIPPET_CHARS = 400
_MAX_FINDINGS = 8
_MAX_HISTORICAL = 5


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _publish_date_sort_key(finding: ScoutFinding) -> datetime:
    """Findings with no/unparseable published_date sort last (oldest), not
    first - Tavily doesn't always populate this field."""
    if not finding.published_date:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = date_parser.parse(finding.published_date)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, OverflowError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _build_context(state: AgentState) -> str:
    lines = [f"# Company: {state['company']}", "", "## Scouted findings"]
    # Selecting the *last-scouted* N findings picks by insertion order across
    # Supervisor loops, not by how current the articles actually are - a
    # fresher result from an earlier loop could get pushed out by a staler
    # one added later. Sort by actual publish date so the Writer always sees
    # the most current findings regardless of scouting order.
    findings = sorted(state["scouted_data"], key=_publish_date_sort_key, reverse=True)[:_MAX_FINDINGS]
    if not findings:
        lines.append("(none)")
    for f in findings:
        date_label = f.published_date or "date unknown"
        lines.append(f"- ({date_label}) [{f.title}]({f.source_url}): {_truncate(f.snippet, _MAX_SNIPPET_CHARS)}")

    lines.append("\n## Historical context")
    historical = state["historical_context"][:_MAX_HISTORICAL]
    if not historical:
        lines.append("(none)")
    for h in historical:
        lines.append(
            f"- (similarity {h.similarity_score:.2f}, sentiment {h.sentiment}): "
            f"{_truncate(h.text, _MAX_SNIPPET_CHARS)}"
        )

    client_context = state.get("client_context") or []
    if client_context:
        lines.append("\n## Client & Competitor Context (from our firm's records, not the public web)")
        for c in client_context:
            if c.client_name == state["company"]:
                detail = f" ({c.client_details})" if c.client_details else ""
                comp_detail = f" ({c.competitor_details})" if c.competitor_details else ""
                text = f"{c.client_name}{detail} is one of our clients. Known competitor: {c.competitor_name}{comp_detail}."
            else:
                detail = f" ({c.competitor_details})" if c.competitor_details else ""
                client_detail = f" ({c.client_details})" if c.client_details else ""
                text = f"{c.competitor_name}{detail} is a known competitor of our client {c.client_name}{client_detail}."
            lines.append(f"- {text}" + (f" Notes: {c.notes}" if c.notes else ""))

    upload = state.get("client_upload_context")
    if upload:
        age_days = (datetime.now(timezone.utc) - upload.uploaded_at).days
        freshness = (
            f"current, uploaded {age_days} day(s) ago"
            if age_days <= CLIENT_REPORT_CACHE_MAX_AGE_DAYS
            else f"stale - uploaded {age_days} days ago, may be out of date"
        )
        lines.append(
            f"\n## Uploaded Client Dossier ({upload.source_filename}, {freshness})\n"
            f"{_truncate(upload.text, _MAX_SNIPPET_CHARS * 6)}"
        )

    if state["sentiment_summary"]:
        lines.append(f"\n## Sentiment summary\n{state['sentiment_summary']}")

    return "\n".join(lines)


async def writer_node(state: AgentState) -> dict:
    # See supervisor.py's comment on method="json_schema"/strict=True - avoids
    # the "Failed to parse tool call arguments as JSON" failures observed
    # under the default best-effort function_calling method.
    llm = get_writer_llm().with_structured_output(WriterOutput, method="json_schema", strict=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    output: WriterOutput = await llm.ainvoke(
        [
            ("system", SYSTEM_PROMPT_TEMPLATE.format(today=today)),
            ("human", _build_context(state)),
        ]
    )

    if output.sufficient_data:
        return {"final_report": output.report_markdown, "insufficient_data_flag": False}

    return {
        "insufficient_data_flag": True,
        "pending_instructions": output.missing_info,
    }
