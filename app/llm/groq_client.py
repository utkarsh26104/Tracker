from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import settings


@lru_cache
def get_supervisor_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_supervisor_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=5,  # langchain-groq's default of 2 isn't enough headroom for
        # free-tier 429s, which can ask for a 10-15s wait under concurrent load
        max_tokens=512,  # routing decisions are short; bounds worst-case TPM usage
    )


@lru_cache
def get_writer_llm() -> ChatGroq:
    return ChatGroq(
        model=settings.groq_writer_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
        max_retries=5,
        # Long, detailed reports can get cut off mid-JSON before the model
        # finishes the WriterOutput tool call, breaking structured-output
        # parsing (observed: groq.BadRequestError "Failed to parse tool call
        # arguments as JSON" on a report that ran past this budget). Capping
        # output forces conciseness and keeps generation inside one response.
        max_tokens=2000,
    )
