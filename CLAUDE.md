# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous multi-agent competitive-intelligence system. Given a list of competitor
companies, a team of LLM agents (LangGraph) researches each one concurrently, drafts an
executive report, pauses for human approval, then synthesizes everything into a Comparison
Matrix. Built on a genuinely free/near-free stack (Groq, Neon Postgres, ChromaDB, Tavily) as a
portfolio/demo project — see `README.md` for the full pitch, cost model, and deployment guide.

## Commands

```bash
# Setup
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -e ".[dev]"
cp .env.example .env                                 # fill in GROQ_API_KEY, TAVILY_API_KEY, DATABASE_URL
python scripts/seed_historical_data.py                # populate ChromaDB with sample history

# Run (two processes, separate terminals)
python scripts/serve.py                               # API on :8000 - see Windows note below
streamlit run ui/streamlit_app.py                      # UI on :8501

# Tests
pytest tests/ -v                       # full suite
pytest tests/test_supervisor.py -v     # single file
pytest tests/ -v --ignore=tests/test_memory.py -k test_name  # single test, skip the slow local-model file
```

No linter/formatter is configured in this repo.

**Windows: always launch the API via `python scripts/serve.py`, never a bare `uvicorn
app.main:app`.** psycopg's async mode (used by the Postgres checkpointer) requires
`SelectorEventLoop`; uvicorn's built-in loop factories return `ProactorEventLoop`
unconditionally on win32 and bypass `asyncio.set_event_loop_policy()` entirely (they pass
`loop_factory=` straight to `asyncio.run()`). `scripts/serve.py` passes the custom factory in
`app/winloop.py` that fixes this — read that file's docstring before touching event-loop code.

Tests skip the Postgres-dependent ones automatically if `DATABASE_URL` isn't reachable (see the
`pg_pool` fixture in `tests/conftest.py`), so the suite runs without secrets configured, just
with less coverage. `tests/test_memory.py` loads real FinancialBERT models with no mocks — can
be slow depending on machine/disk-cache state; unrelated to correctness if it's just slow.

## Architecture

**Map-Reduce over isolated per-company LangGraph state machines.** `POST /tracker/run` (Map)
spawns one independent LangGraph run per company concurrently — no shared context between
companies, each checkpointed to Postgres under its own `thread_id` (`{job_id}:{company}`).
`POST /tracker/approve` (Reduce) resumes only the approved companies past a human-approval gate
and synthesizes their reports into one Comparison Matrix. Companies omitted from `/approve`
just stay paused forever (the reject path — see trade-offs in README).

**The per-company graph** (`app/graph/build_graph.py`) is a cyclic 4-node LangGraph, entirely
`async def` nodes:
- **Supervisor** (`supervisor.py`) — Groq-backed router with a strict Pydantic output schema
  (`RouteDecision`), decides `Search | Analyze | Write | FINISH`. Has a deterministic
  loop-count guardrail that force-terminates without calling the LLM once `max_loops` is hit —
  necessary because a routing model can fail to emit `FINISH` reliably.
- **Scout** (`scout.py`) — Tavily web search, wrapped in `asyncio.to_thread` (sync-only client).
- **Brain** (`brain.py`) — the RAG step, over *two* separate ChromaDB collections. It embeds new
  Scout findings via FinancialBERT and **upserts them into `competitor_history`**
  (`app/memory/vector_store.py`, so the vector store keeps growing across every run, not just
  this session), then queries the same store for similar historical snippets scoped by company
  name, and runs sentiment classification. Separately, it does an exact-match lookup (not a
  similarity search) against `client_context` (`app/memory/client_context.py`) — a
  consultancy-provided CSV of clients and their known competitors (`scripts/
  load_client_context.py`), kept in its own collection so curated firm knowledge doesn't get
  crowded out by web snippets. Both collections get written and read every time Brain runs.
- **Writer** (`writer.py`) — drafts the report from Scout + Brain context (also strict
  structured output, `WriterOutput`), can flag `sufficient_data=false` to force another loop
  instead of fabricating content.
- A **publish_report** node sits behind `interrupt_before=["publish_report"]` — this is the
  HITL gate. The graph genuinely pauses here (verifiable via `graph.aget_state(config).next`)
  until `/tracker/approve` resumes it.

State shape is in `app/graph/state.py` (`AgentState` for the per-company graph channel,
`MasterComparisonState` for the Map-Reduce-level job view). `route_history` on `AgentState` is
the full audit trail of Supervisor decisions — surfaced in the UI as the "agent trace."

**Checkpointing** (`app/db/checkpointer.py`) uses `AsyncPostgresSaver` over a shared
`AsyncConnectionPool`, but **never share one checkpointer instance across concurrently-running
graphs** — `AsyncPostgresSaver` holds an internal `asyncio.Lock` that serializes all cursor
operations *per instance*, so a shared instance silently serializes "concurrent" Map-Reduce
runs regardless of pool size (measured ~10.6s → 4.6s for 3 companies after fixing this). Always
call `make_checkpointer(pool)` fresh per graph run — see that function's docstring.

**Retry/resilience** (`app/graph/master_graph.py`): `_ainvoke_with_retry()` wraps the whole
`graph.ainvoke()` call (not per-node) for transient connection errors — `psycopg.OperationalError`
and `requests.exceptions.ConnectionError` are *not* covered by LangGraph's own node-level
`RetryPolicy` (checkpoint writes happen in LangGraph's Pregel runtime after a node returns,
outside any node's retry boundary; `requests.exceptions.ConnectionError` is an `OSError`
subclass, which LangGraph's default retry predicate explicitly excludes). A retry here is safe
and resumes from the last checkpoint rather than restarting, because the graph is checkpointed.
The same wrapper also catches `groq.RateLimitError` (429s) and parses Groq's own suggested wait
time out of the error message — `ChatGroq`'s built-in `max_retries` (`app/llm/groq_client.py`)
isn't enough on its own under real concurrent Map-Reduce load, since two companies' internal
retries can keep colliding on the same recovering per-minute token budget.

**Structured LLM output**: always pass `method="json_schema", strict=True` to
`.with_structured_output()` for Groq calls (see `supervisor.py`/`writer.py`). The default
`method="function_calling"` is best-effort even on models that support strict mode, and was
observed in real testing to occasionally emit unparseable JSON under load. `strict=True` uses
Groq's actual constrained decoding, which makes malformed output structurally impossible.

**Lazy-loaded singletons need real locks, not just `@lru_cache`**: `@lru_cache` doesn't stop two
concurrent first-callers from both executing the wrapped function body before either has cached
a result — this broke PyTorch model loading (`NotImplementedError: Cannot copy out of meta
tensor`) when two companies' Brain nodes raced to initialize the same FinBERT model
simultaneously via `asyncio.to_thread` (separate OS threads). The lazy singletons in
`app/memory/embeddings.py`, `sentiment.py`, and `vector_store.py` all use manual
double-checked locking (`threading.Lock`) instead.

**Exceptions in Map/Reduce phases must be logged, not just caught.** `run_map_phase` and
`resume_and_finalize` catch per-company exceptions via `asyncio.gather(..., return_exceptions=True)`
and convert them to a `FAILED` status — always pair that with `logger.exception(...,
exc_info=outcome)` (works correctly even outside an active `except` block when the exception
object is passed explicitly), or failures become undiagnosable without manual reproduction.
Similarly, don't let a failure in the *last* step of a multi-step function (e.g. Comparison
Matrix synthesis in `resume_and_finalize`) throw away results that already succeeded earlier in
the same function — wrap it and degrade gracefully instead of crashing the whole request.

**API auth** (`app/api/auth.py`): `require_api_key` is a router-level dependency on the whole
`/tracker` router, checked via `secrets.compare_digest` against `settings.api_key` (`API_KEY` env
var). It's a deliberate no-op when `API_KEY` is unset, so local dev needs no setup — but that
means a production deploy that forgets to set it is silently wide open, which is why
`app/main.py`'s `lifespan` logs a warning at startup when `APP_ENV != "dev"` and no key is
configured. `/health` is intentionally outside the router (unauthenticated), since Render's
health checks and uptime monitors hit it without credentials. The Streamlit UI sends the key via
an `X-API-Key` header (`TRACKER_API_KEY` env var / Streamlit secret) on every request.

**Client report caching** (`master_graph.py`'s `run_map_phase`/`run_company_with_cached_report`,
`app/db/history.py`'s `get_recent_company_report`): a consultancy re-runs this tool repeatedly
for the *same* client against different competitor sets over time — re-researching the client
itself on every run burns Tavily/Groq quota for data that hasn't gone stale.  When
`client_company` names one of the run's companies and `report_history` already has a `company`
report for it younger than `CLIENT_REPORT_CACHE_MAX_AGE_DAYS` (30), that company skips
Scout/Brain/Writer entirely and reuses the cached report — `route_history` gets seeded with
`"Reused cached report..."` as the first entry so the reviewer can see this in the UI (📦 badge)
and the "try again" button (`retry_company`) still works normally if they want a real search
instead. This is implemented by seeding the graph's initial state with `final_report` already
set and `max_loops=0`, which makes the Supervisor's *deterministic* guardrail (not an LLM call)
force `FINISH` on the very first `supervisor_node` invocation — the graph still creates a real
checkpointed thread and pauses at the normal `interrupt_before=["publish_report"]` gate, so
`/tracker/approve`'s `resume_one` needs no special-casing at all. Competitors are never cached
this way, only the named `client_company` — the whole point of a run is fresh intel on
competitors specifically. `CLIENT_REPORT_CACHE_MAX_AGE_DAYS` lives in `app/graph/state.py`, not
`master_graph.py`, because `writer.py` needs it too (see below) and importing it from
`master_graph.py` would create a circular import (`master_graph` → `build_graph` → `writer` →
`master_graph`).

**Client dossier uploads** (`app/memory/client_uploads.py`, `app/services/document_parsing.py`,
`POST /tracker/upload-client-file`): a third ChromaDB collection, separate from
`client_context.py`'s curated CSV roster and `vector_store.py`'s scouted-web-history, for a
consultancy-supplied document (PDF/.txt/.md) about a specific client — text is extracted, chunked
(~1500 chars, paragraph-aware), embedded, and stored keyed by company; re-uploading for the same
company replaces its prior chunks (`collection.delete(where={"company": company})` before
inserting) rather than accumulating stale ones. Brain (`brain.py`) always queries it for whichever
company it's researching (same unconditional-inclusion pattern as `client_context`) and Writer
(`writer.py`) labels it "current" or "stale" in the prompt based on
`CLIENT_REPORT_CACHE_MAX_AGE_DAYS`, so the LLM can judge how much to lean on it versus fresh Scout
findings itself. On top of that, `run_map_phase` gives a **fresh** upload (no existing
system-report cache hit) the same skip-the-web-search treatment as the report cache above: one
direct Writer call over just the dossier (`_draft_report_from_upload`, using `get_writer_llm()`
plainly, not `.with_structured_output`, since there's no `sufficient_data` self-check to make when
Scout never ran) produces a report, which then goes through `run_company_with_cached_report` the
same way a cached system report would. A **stale** upload does *not* get this shortcut — it falls
through to the normal full pipeline, where Brain still includes it (marked stale) alongside real
Scout findings, so the final report blends both rather than ignoring the old dossier outright.

**Seed URLs for thin-coverage companies** (`app/services/scraping.py`'s
`fetch_company_site_findings`/`fetch_url_as_finding`, `run_company`'s `seed_url` param,
`RunRequest.company_urls`): observed in real use — small/local/private businesses with little
press coverage (e.g. `newme`, `bonkers`) can drive the Supervisor through every one of its
`max_loops` Search iterations without ever finding enough to write from, then fail outright on
the final forced Write once that also collides with Groq's rate limit (see below). The UI's
Competitors textarea accepts an optional `Name | https://url` per line; `fetch_company_site_findings`
fetches that URL, then does a light same-domain crawl (capped at `_MAX_SEED_PAGES`, default 4)
following links whose href/text match product-, discount-, or review-related keywords
(`_RELEVANT_LINK_KEYWORDS`) — deliberately *not* a general-purpose spider, just enough to surface
launches/pricing/discounts and customer reviews for a company that generic web search can't find
anything about. Each page's text extraction (`_extract_finding`, shared by both functions) strips
`<script>/<style>/<nav>/<footer>/<header>` and scopes `get_text()` to `<body>` specifically —
extracting from the whole document would leak `<title>` text into the snippet and could even make
a genuinely empty page look non-empty. Review-page text isn't given special rating-extraction
logic — it becomes an ordinary `ScoutFinding` like any other, so it flows through Brain's existing
sentiment pipeline (`summarize_sentiment` on every scouted snippet) automatically, and the Writer
is trusted to pull a mentioned rating out of messy page text directly rather than a fragile regex
trying to do it upstream. All fetched findings seed `scouted_data` *before* the graph runs, with
`route_history` recording `"Seeded with N page(s) from provided URL: ..."` as the first entry. A
failed fetch (bad URL, network error, empty homepage) returns `[]`/`None` and is logged, not
raised — Scout still runs normally either way, this is purely additive. Scoped to competitors
only (via the textarea); the client already has the richer dossier-upload path above, which fully
replaces the need for this on the client specifically.

**Amazon/Flipkart marketplace search** (`app/services/scraping.py`'s `search_marketplace_reviews`,
called alongside `fetch_company_site_findings` whenever a seed URL is given — see `run_company`):
found in real use that a company's own site is often JS-rendered (a plain `httpx` fetch sees an
empty shell, no real product/review content at all), and a company's own site rarely has honest
customer sentiment anyway. Rather than writing a direct Amazon/Flipkart scraper — both are
aggressively bot-hostile (CAPTCHAs, rate limiting) and would likely just get blocked — this reuses
the *existing* Tavily integration with `include_domains=["amazon.in", "amazon.com",
"flipkart.com"]`, which restricts results server-side to pages Tavily's own crawler already
indexed. Runs concurrently with the site crawl (`asyncio.gather`) since both are independent
`asyncio.to_thread` calls. Only triggered when a seed URL is provided — same opt-in signal as the
site crawl, not a blanket extra Tavily call on every company.

**Forced report on the final loop for a seeded company** (`app/graph/state.py`'s
`AgentState.seeded_from_url`, `writer.py`'s `FORCED_FINAL_ATTEMPT_INSTRUCTION`/`force_report`):
even with seeded data, the Writer can still self-assess `sufficient_data=false` and bounce the
Supervisor back into another Search loop — for a company that was seeded specifically *because*
Tavily has nothing for it, those extra loops just burn Groq calls (and rate-limit exposure) on
searches that were never going to find anything, right up until the loop cap forces one last
Write anyway. `run_company` sets `seeded_from_url=True` on the initial state whenever
`fetch_company_site_findings`/`search_marketplace_reviews` returned anything; `writer_node`
checks `state["loop_count"] > state["max_loops"]` (the exact condition `supervisor_node`'s own
guardrail uses to force this Write in the first place) together with that flag, and if both hold,
appends an instruction telling the Writer this is the last chance and it must publish *something*
from what's available rather than declining again. Since an LLM won't follow that instruction
with certainty, `writer_node` also has a deterministic backstop: it accepts `report_markdown`
in this specific case even if `sufficient_data` still comes back `false`, as long as the
markdown isn't empty — same "don't just trust the model under pressure" principle as the
Supervisor's own loop-count guardrail. A truly empty response still correctly falls through to
`insufficient_data_flag=True` rather than publishing nothing dressed up as a report.

**UI request timeout vs. real worst-case run time**: `call_api_with_progress`'s `httpx.request`
timeout is 600s, not the more obvious-looking 300s — a multi-company run's worst case isn't one
slow LLM call, it's `(loop cap) × (per-loop work + rate-limit retries)`, and companies running
*concurrently* share Groq's per-minute token budget, so several of them hitting that worst case
at once (as happened with the thin-coverage case above) is a real scenario, not a hypothetical
one. 300s was observed too tight for it in practice.

**Rate-limit wait parsing had a real bug, and long waits shouldn't be slept through anyway**
(`master_graph.py`'s `_RETRY_AFTER_RE`/`_rate_limit_wait_seconds`/`describe_exception`): Groq's
wait-time format differs by which quota was hit — per-minute limits say e.g. `"try again in
18.5475s"`, but the *daily* (TPD) quota says e.g. `"try again in 16m32.304s"`. The original regex
only captured a bare seconds group, so it silently failed to match the minutes-containing format
at all and fell back to a far-too-short guess (`5.0 * attempt`) — meaning a run that hit the daily
cap kept retrying every few seconds instead of actually waiting the ~16 minutes Groq asked for,
burning through `max_attempts` uselessly. Fixed the regex to capture an optional minutes group.
Separately, even with correct parsing, sleeping through a many-minutes wait inside a live request
is pointless — `_MAX_RATE_LIMIT_SLEEP_SECONDS` (60s) caps how long `_ainvoke_with_retry` will
actually sleep for; past that it fails fast instead. `describe_exception(exc)` turns whatever
killed a company (rate limit, transient connection error, or anything else) into a short,
user-facing message — including Groq's *actual* reported wait time for rate limits specifically,
since "no report was produced (insufficient data or an error)" doesn't tell a reviewer whether to
retry in 20 seconds or 20 minutes. Surfaced end-to-end via `MasterComparisonState.company_errors`
→ `RunResponse.company_errors` / `RetryResponse.error` → the review page's per-company warning.
