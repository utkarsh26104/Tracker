import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    """Gate for /tracker/*. No-op when API_KEY isn't configured (local dev) -
    once set (e.g. before a public deploy), every request must present a
    matching X-API-Key header, otherwise a stranger with the URL could burn
    through the Groq/Tavily quotas."""
    if not settings.api_key:
        return
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
