"""In-process registry of Gemini explicit context caches (`CachedContent`)
for agent 1's static prompt block. See src/utils/gemini_cache.py for the
original — this is the definitive location; the original is a backward-compat
re-export.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.utils.logging_utils import console_log


@dataclass
class _CacheEntry:
    cached_content: Any
    expires_at: datetime


_REGISTRY: dict[str, _CacheEntry] = {}

_EXPIRY_SAFETY_MARGIN = timedelta(seconds=60)


def _cache_key(mode: str, model_name: str, system_instruction: str, static_context: str) -> str:
    digest = hashlib.sha256(f"{system_instruction}\x00{static_context}".encode("utf-8")).hexdigest()[:16]
    return f"{mode}:{model_name}:{digest}"


def get_or_create_cache(
    *,
    mode: str,
    model_name: str,
    system_instruction: str,
    static_context: str,
    ttl_seconds: int,
) -> Optional[Any]:
    """Returns a live CachedContent or None on failure (graceful degradation)."""
    key = _cache_key(mode, model_name, system_instruction, static_context)
    now = datetime.now(timezone.utc)

    entry = _REGISTRY.get(key)
    if entry is not None and entry.expires_at - now > _EXPIRY_SAFETY_MARGIN:
        return entry.cached_content

    from google.generativeai import caching

    qualified_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
    try:
        cached_content = caching.CachedContent.create(
            model=qualified_model,
            display_name=key,
            system_instruction=system_instruction,
            contents=[static_context],
            ttl=timedelta(seconds=ttl_seconds),
        )
    except Exception as exc:  # noqa: BLE001
        console_log(
            mode,
            "GEMINI_CACHE",
            f"cache create failed for key={key!r} (falling back to inline static context): {exc}",
        )
        return None

    _REGISTRY[key] = _CacheEntry(cached_content=cached_content, expires_at=now + timedelta(seconds=ttl_seconds))
    console_log(
        mode,
        "GEMINI_CACHE",
        f"created cache key={key!r} ttl={ttl_seconds}s name={getattr(cached_content, 'name', '?')}",
    )
    return cached_content
