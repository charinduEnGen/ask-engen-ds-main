"""LLM configuration endpoints — called by the backend to hold API keys in
memory and push config changes at save time.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.llm import store as llm_store

router = APIRouter(prefix="/llm", tags=["llm"])

# Backend service IDs that map to the DS "analytics" mode
_SERVICE_ID_MAP: dict[str, str] = {
    "datapulse": "analytics",
    "analytics": "analytics",
}


def _resolve(service_id: str) -> str:
    return _SERVICE_ID_MAP.get(service_id, service_id)


class TestKeyRequest(BaseModel):
    service_id: str
    provider: str
    api_key: str
    hold_in_memory: bool = True


class ConfigRequest(BaseModel):
    service_id: str
    mode: str
    status: str
    primary_provider: Optional[str] = None
    api_key: Optional[str] = None
    provider: Optional[str] = None
    public_pool_order: list[str] = []
    updated_at: Optional[str] = None


def _validate_gemini_key(api_key: str) -> None:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    # list_models is a lightweight auth-only round-trip (no tokens consumed)
    list(genai.list_models())


@router.post("/test-key")
async def test_key(body: TestKeyRequest):
    if body.provider != "gemini":
        raise HTTPException(
            status_code=400,
            detail={"message": f"Provider '{body.provider}' is not supported in phase 1"},
        )

    try:
        await asyncio.to_thread(_validate_gemini_key, body.api_key)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={"message": f"Key validation failed: {exc}"},
        ) from exc

    if body.hold_in_memory:
        llm_store.set_key(_resolve(body.service_id), body.provider, body.api_key)

    return {"success": True, "message": "Key validated"}


@router.post("/config")
async def set_config(body: ConfigRequest):
    service_id = _resolve(body.service_id)
    provider = body.provider or body.primary_provider

    llm_store.set_config(
        service_id=service_id,
        provider=provider,
        api_key=body.api_key,
        mode=body.mode,
    )

    return {"success": True, "message": "Config updated"}
