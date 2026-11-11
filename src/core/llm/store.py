"""In-memory LLM configuration store.

Receives provider + API key from the backend via /llm/test-key and /llm/config.
call_llm() reads from here first, falling back to .env settings.
"""
import threading
from typing import Optional

_lock = threading.Lock()
# service_id -> {provider, api_key, mode}
_configs: dict[str, dict] = {}


def set_key(service_id: str, provider: str, api_key: str) -> None:
    with _lock:
        entry = _configs.setdefault(service_id, {})
        entry["provider"] = provider
        entry["api_key"] = api_key


def set_config(
    service_id: str,
    provider: Optional[str],
    api_key: Optional[str],
    mode: Optional[str],
) -> None:
    with _lock:
        entry = _configs.setdefault(service_id, {})
        if provider:
            entry["provider"] = provider
        if api_key:
            entry["api_key"] = api_key
        if mode:
            entry["mode"] = mode


def get_key(service_id: str, provider: str) -> Optional[str]:
    with _lock:
        entry = _configs.get(service_id, {})
        if entry.get("provider") == provider:
            return entry.get("api_key")
        return None
