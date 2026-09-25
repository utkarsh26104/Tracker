import os
import sys
import threading
import time
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.report import markdown_to_pdf  # noqa: E402

try:
    # Streamlit Community Cloud's Secrets panel populates st.secrets, not
    # os.environ, directly - bridge it so TRACKER_API_URL/TRACKER_API_KEY
    # work the same way there as they do locally via .env.
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass  # no secrets.toml (e.g. local dev) - .env/shell env vars cover that case

API_BASE_URL = os.environ.get("TRACKER_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("TRACKER_API_KEY", "")
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Tracker - Competitive Intelligence", layout="wide")

if "phase" not in st.session_state:
    st.session_state.phase = "input"  # input -> review -> final; history is a side branch
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "map_result" not in st.session_state:
    st.session_state.map_result = None
if "final_result" not in st.session_state:
    st.session_state.final_result = None
if "history_data" not in st.session_state:
    st.session_state.history_data = None
if "history_from_phase" not in st.session_state:
    st.session_state.history_from_phase = "input"
if "client_company" not in st.session_state:
    st.session_state.client_company = ""
if "report_chat_history" not in st.session_state:
    st.session_state.report_chat_history = []

with st.sidebar:
    st.title("Tracker")
    st.caption("Autonomous multi-agent competitive intelligence")
    st.markdown(
        """
**Architecture**

- **Supervisor** (Groq gpt-oss-120b) routes between agents
- **Scout** pulls live web data via Tavily
- **Brain** embeds + retrieves history with local FinancialBERT + ChromaDB
- **Writer** (Groq gpt-oss-120b) drafts the report
- A **human approval gate** sits before anything is published

Each company runs as an isolated, concurrent LangGraph run,
checkpointed to Postgres so nothing is lost mid-run.
        """
    )
    st.divider()
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Start over"):
        st.session_state.phase = "input"
        st.session_state.job_id = None
        st.session_state.map_result = None
        st.session_state.final_result = None
        st.session_state.client_company = ""
        st.session_state.report_chat_history = []
        st.rerun()

st.title("Competitive Landscape Tracker")

STATUS_EMOJI = {
    "awaiting_approval": "🟡",
    "done": "🟢",
    "failed": "🔴",
    "running": "⏳",
}


def render_route_history(history: list[str]) -> None:
    if not history:
        st.caption("No agent trace recorded.")
        return
    st.write(" → ".join(history))


def render_report(content: str) -> None:
    """Renders LLM-generated report markdown. unsafe_allow_html is needed
    because standard Markdown tables can't contain real line breaks within a
    cell - the Writer uses inline <br> tags for multi-line cells (e.g. a
    bulleted list inside one table cell), which st.markdown() otherwise shows
    as literal text instead of rendering. Safe here: this is our own LLM's
    structured output, not third-party user-submitted content."""
    st.markdown(content, unsafe_allow_html=True)


def call_api_with_progress(method: str, url: str, json: dict, running_label: str, messages: list[str]):
    """Runs a (slow, synchronous) API call in a background thread while the
    main thread keeps the UI responsive with a rotating status log - the API
    call itself is one blocking HTTP request with no intermediate progress to
    report, so these messages describe what's *likely* happening agent-side
    rather than tracking real state."""
    result: dict = {}

    def _do_request():
        try:
            # A multi-company run's worst case isn't bounded by a single LLM
            # call - it's (loop cap) x (per-loop work + rate-limit retries),
            # and concurrent companies sharing Groq's per-minute token budget
            # can push several of them into that worst case at once. 300s
            # was observed to be too tight for that combination in practice;
            # 600s gives real (slow but legitimate) runs room to finish
            # instead of the UI reporting "timed out" on top of - or instead
            # of - the backend's own per-company FAILED status.
            resp = httpx.request(method, url, json=json, headers=API_HEADERS, timeout=600.0)
            resp.raise_for_status()
            result["data"] = resp.json()
        except httpx.HTTPError as e:
            result["error"] = str(e)

    thread = threading.Thread(target=_do_request)
    thread.start()

    with st.status(running_label, expanded=True) as status_box:
        i = 0
        elapsed = 0
        while thread.is_alive():
            status_box.write(f"[{elapsed}s] {messages[i % len(messages)]}")
            time.sleep(4)
            elapsed += 4
            i += 1
        thread.join()

        if "error" in result:
            status_box.update(label="Request failed", state="error")
            return None, result["error"]

        status_box.update(label="Done", state="complete")
        return result["data"], None


RESEARCH_MESSAGES = [
    "Scout agents are searching the web for recent news and pricing changes...",
    "Brain agents are embedding findings and retrieving historical context...",
    "Writer agents are drafting executive reports...",
    "Supervisor agents are checking whether more research is needed...",
    "Still working - real web research and LLM reasoning takes a little while...",
]

PUBLISH_MESSAGES = [
    "Resuming approved companies past the human-approval gate...",
    "Synthesizing individual reports into the Comparison Matrix...",
    "Almost done...",
]


RECENCY_OPTIONS = {
    "Last 7 days": 7,
    "Last 15 days": 15,
    "Last 1 month": 30,
}

if st.session_state.phase == "input":
    client_company_raw = st.text_input(
        "Your client's company name (optional)",
        value=st.session_state.client_company,
        placeholder="e.g. Acme Retail, or Acme Retail | https://acmeretail.com",
        help="If set, the final report becomes a Strategic Recommendation for this company - "
        "comparing it against the others and recommending moves in response to their activity - "
        "instead of a neutral comparison. This company is researched too even if you don't also "
        "list it below. If the client has little web coverage (small, local, or private "
        "business), add its website after a `|` the same way you would for a competitor below - "
        "it gets the same site crawl + Amazon/Flipkart/Instagram treatment.",
    )
    if "|" in client_company_raw:
        client_name, client_url = (part.strip() for part in client_company_raw.split("|", 1))
        client_url = client_url or None
    else:
        client_name, client_url = client_company_raw.strip(), None

    if client_name:
        with st.expander("📎 Upload a client dossier (optional)", expanded=False):
            st.caption(
                "Already have a profile on this client? Upload it (PDF/.txt/.md) and future runs "
                "will reuse it instead of doing a fresh web search, as long as it's still recent "
                "enough (< 30 days). A stale upload still gets blended with fresh search results "
                "rather than ignored."
            )
            dossier_file = st.file_uploader(
                "Client dossier", type=["pdf", "txt", "md"], key="dossier_uploader", label_visibility="collapsed"
            )
            if st.button("Upload Dossier", disabled=dossier_file is None):
                try:
                    resp = httpx.post(
                        f"{API_BASE_URL}/tracker/upload-client-file",
                        data={"company": client_name},
                        files={"file": (dossier_file.name, dossier_file.getvalue())},
                        headers=API_HEADERS,
                        timeout=60.0,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"Uploaded {result['source_filename']} for {result['company']} "
                        f"({result['chunk_count']} chunk(s))."
                    )
                except httpx.HTTPError as e:
                    st.error(f"Upload failed: {e}")

    st.write("Enter one competitor per line, then run the research agents.")
    raw = st.text_area(
        "Competitors",
        value="Stripe\nAdyen",
        height=120,
        help="One per line. If a competitor has little web coverage (e.g. a small, local, or "
        "private business), add its own website after a `|` to seed the research directly from "
        "it - e.g. `LocalBrand | https://localbrand.com`. This also crawls that site for "
        "product/launch and discount pages, and searches Amazon/Flipkart/Instagram/Facebook for "
        "the company's own listings, posts, and customer reviews - so sentiment analysis has "
        "something real to work with even when the company's site is JS-rendered or has no "
        "reviews of its own. Otherwise a thin-coverage company can burn through several fruitless "
        "search loops before giving up.",
    )
    recency_label = st.selectbox(
        "How far back should Scout search?",
        options=list(RECENCY_OPTIONS.keys()),
        index=2,  # default: Last 1 month
        help="Narrower windows return fresher results but less coverage for less-followed companies.",
    )
    search_days = RECENCY_OPTIONS[recency_label]

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Run Research", type="primary"):
            companies = []
            company_urls = {}
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "|" in line:
                    name, url = (part.strip() for part in line.split("|", 1))
                    if name:
                        companies.append(name)
                        if url:
                            company_urls[name] = url
                else:
                    companies.append(line)

            if client_name and client_url:
                company_urls[client_name] = client_url

            if not companies and not client_name:
                st.error("Enter at least one company.")
            else:
                st.session_state.client_company = client_name
                data, error = call_api_with_progress(
                    "POST",
                    f"{API_BASE_URL}/tracker/run",
                    {
                        "companies": companies,
                        "search_days": search_days,
                        "client_company": st.session_state.client_company or None,
                        "company_urls": company_urls,
                    },
                    f"Researching {len(companies)} companies concurrently ({recency_label.lower()})...",
                    RESEARCH_MESSAGES,
                )
                if error:
                    st.error(f"Request failed: {error}")
                else:
                    st.session_state.map_result = data
                    st.session_state.job_id = data["job_id"]
                    st.session_state.phase = "review"
                    st.rerun()
    with col2:
        if st.button("View Report History"):
            try:
                resp = httpx.get(f"{API_BASE_URL}/tracker/history", headers=API_HEADERS, timeout=30.0)
                resp.raise_for_status()
                st.session_state.history_data = resp.json()["entries"]
                st.session_state.history_from_phase = "input"
                st.session_state.phase = "history"
                st.rerun()
            except httpx.HTTPError as e:
                st.error(f"Could not load history: {e}")

elif st.session_state.phase == "review":
    result = st.session_state.map_result
    st.subheader("Review draft reports")
    if st.session_state.client_company:
        st.caption(
            f"Approve the companies to include. Final report will be a Strategic Recommendation "
            f"for **{st.session_state.client_company}**, not a neutral comparison."
        )
    else:
        st.caption("Approve the companies you want included in the final Comparison Matrix.")

    approved = []
    for company, status in result["company_statuses"].items():
        emoji = STATUS_EMOJI.get(status, "⚪")
        label = f"{emoji} {company} - {status}"
        if company == st.session_state.client_company:
            label += " 👑 (your client)"
        route_history = result["company_route_histories"].get(company, [])
        if route_history and route_history[0].startswith("Reused cached report"):
            label += " 📦 (reused recent data, not freshly searched)"
        with st.expander(label, expanded=True):
            if company in result["company_reports"]:
                checked = st.checkbox("Approve for publication", value=True, key=f"approve_{company}")
                if checked:
                    approved.append(company)
                render_report(result["company_reports"][company])
                with st.popover("Agent trace"):
                    render_route_history(result["company_route_histories"].get(company, []))
            else:
                error_msg = result.get("company_errors", {}).get(company)
                st.warning(error_msg or "No report was produced for this company (insufficient data or an error).")

            if st.button(f"🔄 Try again for {company}", key=f"retry_{company}"):
                data, error = call_api_with_progress(
                    "POST",
                    f"{API_BASE_URL}/tracker/retry",
                    {"job_id": st.session_state.job_id, "company": company},
                    f"Re-researching {company}...",
                    RESEARCH_MESSAGES,
                )
                if error:
                    st.error(f"Retry failed: {error}")
                else:
                    # Merge just this company's fresh result back into the
                    # existing map_result - siblings are untouched.
                    result["company_statuses"][company] = data["status"]
                    result["company_route_histories"][company] = data["route_history"]
                    result.setdefault("company_errors", {})
                    if data["report"]:
                        result["company_reports"][company] = data["report"]
                        result["company_errors"].pop(company, None)
                    else:
                        result["company_reports"].pop(company, None)
                        if data.get("error"):
                            result["company_errors"][company] = data["error"]
                    st.session_state.map_result = result
                    st.rerun()

    st.divider()
    if st.session_state.client_company and st.session_state.client_company not in approved:
        st.info(
            f"Your client, {st.session_state.client_company}, isn't approved - the final report "
            "will fall back to a neutral Comparison Matrix instead of a strategy for them."
        )
    if st.button("Approve & Publish", type="primary", disabled=not approved):
        data, error = call_api_with_progress(
            "POST",
            f"{API_BASE_URL}/tracker/approve",
            {
                "job_id": st.session_state.job_id,
                "companies": approved,
                "client_company": st.session_state.client_company or None,
            },
            "Publishing approved reports and building the comparison matrix...",
            PUBLISH_MESSAGES,
        )
        if error:
            st.error(f"Request failed: {error}")
        else:
            st.session_state.final_result = data
            st.session_state.report_chat_history = []
            st.session_state.phase = "final"
            st.rerun()

elif st.session_state.phase == "final":
    result = st.session_state.final_result
    is_strategy = bool(st.session_state.client_company) and st.session_state.client_company in result.get(
        "company_reports", {}
    )
    if is_strategy:
        st.subheader(f"Strategic Recommendation for {st.session_state.client_company}")
        file_prefix = f"strategy_{st.session_state.client_company}"
    else:
        st.subheader("Competitive Landscape Comparison Matrix")
        file_prefix = "comparison_matrix"

    if result["comparison_matrix"]:
        render_report(result["comparison_matrix"])

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download report (Markdown)",
                data=result["comparison_matrix"],
                file_name=f"{file_prefix}_{st.session_state.job_id}.md",
                mime="text/markdown",
            )
        with col2:
            st.download_button(
                "Download report (PDF)",
                data=markdown_to_pdf(result["comparison_matrix"]),
                file_name=f"{file_prefix}_{st.session_state.job_id}.pdf",
                mime="application/pdf",
            )
    else:
        st.warning("No comparison matrix was produced.")

    if result.get("comparison_matrix") or result.get("company_reports"):
        st.divider()
        st.subheader("💬 Ask about this report")
        st.caption(
            "Ask questions about the report and strategy above - answers are grounded only in "
            "this content, not a fresh web search."
        )

        for turn in st.session_state.report_chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        question = st.chat_input("e.g. Why is pricing flagged as a risk?")
        if question:
            st.session_state.report_chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Full context: the top-level comparison/strategy plus every
            # individual company report, so questions about either the
            # overall strategy or one company's specifics can be answered.
            context_parts = []
            if result.get("comparison_matrix"):
                heading = (
                    f"Strategic Recommendation for {st.session_state.client_company}"
                    if is_strategy
                    else "Comparison Matrix"
                )
                context_parts.append(f"# {heading}\n{result['comparison_matrix']}")
            for company, report in result.get("company_reports", {}).items():
                context_parts.append(f"# {company} Report\n{report}")
            report_context = "\n\n".join(context_parts)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        resp = httpx.post(
                            f"{API_BASE_URL}/tracker/ask",
                            json={
                                "report_context": report_context,
                                "question": question,
                                "history": st.session_state.report_chat_history[:-1],
                            },
                            headers=API_HEADERS,
                            timeout=60.0,
                        )
                        resp.raise_for_status()
                        answer = resp.json()["answer"]
                    except httpx.HTTPError as e:
                        answer = f"Sorry, something went wrong: {e}"
                    st.markdown(answer)
            st.session_state.report_chat_history.append({"role": "assistant", "content": answer})

    st.divider()
    st.caption("Individual reports")
    for company, report in result["company_reports"].items():
        with st.expander(company):
            render_report(report)
            st.download_button(
                "Download (PDF)",
                data=markdown_to_pdf(report),
                file_name=f"{company}_{st.session_state.job_id}.pdf",
                mime="application/pdf",
                key=f"pdf_{company}",
            )

elif st.session_state.phase == "history":
    st.subheader("Report History")
    st.caption("Every report that has been approved and published so far, newest first.")

    if st.button("← Back"):
        st.session_state.phase = st.session_state.history_from_phase
        st.rerun()

    entries = st.session_state.history_data or []
    if not entries:
        st.info("No published reports yet - run some research and approve it to see history here.")
    else:
        for i, entry in enumerate(entries):
            if entry["report_type"] == "comparison_matrix":
                label = f"📊 Comparison Matrix — {entry['created_at']} (job {entry['job_id'][:8]})"
            elif entry["report_type"] == "strategy":
                label = f"🎯 Strategy for {entry['company']} — {entry['created_at']} (job {entry['job_id'][:8]})"
            else:
                label = f"🏢 {entry['company']} — {entry['created_at']} (job {entry['job_id'][:8]})"

            with st.expander(label):
                render_report(entry["content"])
                st.download_button(
                    "Download (PDF)",
                    data=markdown_to_pdf(entry["content"]),
                    file_name=f"{entry['report_type']}_{entry['job_id']}.pdf",
                    mime="application/pdf",
                    key=f"history_pdf_{i}",
                )
