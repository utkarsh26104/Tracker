# Cost model

The architecture's core bet: split cheap, high-volume work (embeddings, retrieval) from
expensive, low-volume reasoning (routing, writing), and put each on the tier that fits.

## Per-component pricing

| Component | Service | Price |
|---|---|---|
| Supervisor (routing) | Groq `openai/gpt-oss-120b` | $0 on free tier (rate-limited: ~1,000 req/day, ~8K tokens/min, org-wide) |
| Writer (report drafting) | Groq `openai/gpt-oss-120b` | $0 on free tier, same limits |
| Embeddings | FinancialBERT, local CPU (HuggingFace `transformers`) | $0 - no API call |
| Sentiment classification | FinBERT, local CPU | $0 - no API call |
| Vector storage/retrieval | ChromaDB, embedded | $0 - no hosted service |
| Web search | Tavily free tier | $0 up to 1,000 credits/month |

Every component in this stack runs at genuinely **$0** at prototype/demo scale - no metered
billing anywhere, only rate limits. That's a deliberate choice: the original design (see
`hybrid_architecture_tracker.pdf`) called for Amazon Bedrock's paid-but-cheap models for the
Supervisor/Writer roles; in practice new AWS accounts hit a quota catch-22 that blocks any usage
at all (see the README's "Why Groq and not Bedrock?" section), so the LLM layer moved to Groq's
free tier instead - same architecture, same `.with_structured_output()` interface, zero billing
risk during development.

## Why the LLM/SLM split still matters even at $0

Even with Groq's free tier removing per-token *billing* as a concern, the split between cheap
high-volume work and expensive low-volume reasoning still matters because Groq's free tier is
**rate-limited**, not unlimited: ~1,000 requests/day, org-wide, for `gpt-oss-120b`. A Supervisor
loop makes several LLM calls per company per run; embeddings and sentiment scoring happen on
*every* scouted document - by far the highest-volume workload in this pipeline. Routing those
through the same rate-limited free-tier LLM would exhaust the daily quota almost immediately.
Keeping embeddings/sentiment on a local, unlimited model is what makes the LLM-call budget
actually sustainable for anything beyond a single demo run.

## Illustrative comparison: this architecture vs. an all-OpenAI-embeddings baseline

Assume a workload of 1,000,000 documents/month needing embedding (a realistic volume for an
enterprise competitive-intelligence tool ingesting continuous news/pricing data across many
tracked competitors), each ~500 tokens:

- **OpenAI `text-embedding-3-small`** at $0.02 / 1M tokens: 1,000,000 × 500 tokens = 500M
  tokens → **~$10/month** for embeddings alone, scaling linearly with volume and requiring a
  live API dependency for every single document.
- **Local FinancialBERT (this architecture)**: **$0/month** in API cost at any volume - the
  only cost is the compute already needed to run the app (CPU inference, no GPU required at
  this model size).

At this scale the absolute dollar saving is modest, but the ratio - **100% reduction in
per-document embedding cost** - holds at any volume, and the local model is
tuned for financial-domain semantics, which a generic embedding API isn't. The bigger point for
a production deployment isn't the $10 - it's that per-document cost stays at exactly $0 as
ingestion volume grows, while a metered embeddings API's cost grows linearly forever. The same
logic is why a production deployment scaling past Groq's free-tier rate limits would want either
a paid Groq tier or Bedrock's pay-per-token model (both integrate the same way, per
`app/llm/groq_client.py`'s interface) rather than moving embeddings to a metered API too.

## Actual measured cost for this prototype

Every environment/integration test in this build ran against mocked LLM/Tavily calls
specifically to keep real spend and rate-limit usage near zero during development (see the
build log in the main README). The only real costs incurred building this were:
- Neon Postgres: $0 (free tier)
- Tavily: $0 (free tier, well under the 1,000 credit/month cap)
- Groq: $0 (free tier, no billing at all - only rate limits)
