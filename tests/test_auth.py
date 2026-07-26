import pytest
from fastapi import HTTPException

from app.api.auth import require_api_key
from app.config import settings


@pytest.mark.asyncio
async def test_no_api_key_configured_allows_any_request(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "")
    await require_api_key(provided=None)


@pytest.mark.asyncio
async def test_missing_header_rejected_once_api_key_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(provided=None)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_wrong_key_rejected(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(provided="wrong")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_correct_key_allowed(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret123")
    await require_api_key(provided="secret123")
