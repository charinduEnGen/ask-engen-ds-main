"""LLM configuration endpoints — called by the backend to hold API keys in
memory and push config changes at save time.
"""
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.llm import store as llm_store

# ---------------------------------------------------------------------------
# Display metadata for the public-mode OpenRouter pool.
# Keyed by the exact model ID in PUBLIC_MODEL_POOL.
# ---------------------------------------------------------------------------
_PUBLIC_MODEL_DISPLAY: dict[str, dict] = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": {
        "name": "NVIDIA Nemotron Ultra 550B",
        "provider_name": "NVIDIA",
        "provider_key": "nvidia",
        "latency": "~1200ms",
        "latency_ms": 1200,
        "released": "Jun 2025",
    },
    "nvidia/nemotron-3.5-lightning:free": {
        "name": "NVIDIA Nemotron 3.5 Lightning",
        "provider_name": "NVIDIA",
        "provider_key": "nvidia",
        "latency": "~400ms",
        "latency_ms": 400,
        "released": "Jul 2025",
    },
}


def _fmt_tokens(n: int) -> str:
    """Format a token count as a human-readable string (e.g. 613M, 3.46T)."""
    if n == 0:
        return "0"
    if n >= 1_000_000_000_000:
        v = n / 1_000_000_000_000
        return f"{v:.2f}".rstrip("0").rstrip(".") + "T"
    if n >= 1_000_000_000:
        v = n / 1_000_000_000
        return f"{v:.2f}".rstrip("0").rstrip(".") + "B"
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:.1f}".rstrip("0").rstrip(".") + "K"
    return str(n)


def _fmt_context(n: int) -> str:
    """Format a context window size (e.g. 262144 → '262K', 1048576 → '1M')."""
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)

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


@router.get("/public-usage")
async def get_public_usage():
    """Return the current-week token usage for every model in the public pool.

    Called by the backend to populate the LLM Configuration page's public-mode
    table with live progress bars.
    """
    from src.core.llm import token_tracker

    active_model = await asyncio.to_thread(token_tracker.get_active_model)
    models = []
    for m in token_tracker.PUBLIC_MODEL_POOL:
        model_id = m["id"]
        weekly_limit = m["weekly_limit"]
        context_window = m["context_window"]
        tokens_used = await asyncio.to_thread(token_tracker.get_weekly_usage, model_id)
        percentage = round(tokens_used / weekly_limit * 100, 2) if weekly_limit else 0.0
        over = await asyncio.to_thread(token_tracker.is_over_threshold, model_id, weekly_limit)
        display = _PUBLIC_MODEL_DISPLAY.get(model_id, {})
        models.append({
            "id": model_id,
            "name": display.get("name", model_id),
            "provider_name": display.get("provider_name", ""),
            "provider_key": display.get("provider_key", ""),
            "weekly_limit": weekly_limit,
            "weekly_limit_label": _fmt_tokens(weekly_limit),
            "context_window": context_window,
            "context_label": _fmt_context(context_window),
            "tokens_used": tokens_used,
            "tokens_used_label": _fmt_tokens(tokens_used),
            "percentage": percentage,
            "is_active": model_id == active_model,
            "over_threshold": over,
            "latency": display.get("latency", ""),
            "latency_ms": display.get("latency_ms", 0),
            "released": display.get("released", ""),
        })
    return {"models": models, "active_model": active_model}
