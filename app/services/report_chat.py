"""Q&A over an already-generated report/strategy. Grounded directly in the
report text the caller supplies (not a fresh RAG lookup) - the whole point
is answering questions about *this* report, which the user already has in
front of them, not re-researching the company."""

from app.llm.groq_client import get_writer_llm

SYSTEM_PROMPT_TEMPLATE = """You are answering follow-up questions about a competitive-intelligence
report/strategy that has already been generated and is shown to the user below. Base every answer
strictly on this content - if something wasn't covered in the report, say so plainly rather than
guessing or inventing details that aren't there.

Keep answers concise (a few sentences, or a short list) unless the question calls for more detail.

# Report content
{report_context}"""

# Groq's TPM budget is shared with every other call in this app - cap both
# the report context and how much conversation history rides along, so a
# long chat session or a large multi-company report can't blow the budget.
_MAX_CONTEXT_CHARS = 12000
_MAX_HISTORY_TURNS = 6

_ROLE_MAP = {"user": "human", "assistant": "ai"}


async def answer_report_question(report_context: str, question: str, history: list[dict]) -> str:
    if len(report_context) > _MAX_CONTEXT_CHARS:
        report_context = report_context[:_MAX_CONTEXT_CHARS].rstrip() + "..."

    llm = get_writer_llm()
    messages = [("system", SYSTEM_PROMPT_TEMPLATE.format(report_context=report_context))]
    for turn in history[-_MAX_HISTORY_TURNS:]:
        messages.append((_ROLE_MAP.get(turn["role"], turn["role"]), turn["content"]))
    messages.append(("human", question))

    response = await llm.ainvoke(messages)
    return response.content
