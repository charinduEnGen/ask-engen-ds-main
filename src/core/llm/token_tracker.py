"""Weekly token tracker for public-mode OpenRouter model rotation.

The pool and limits are owned by DS — not received from the backend.
get_active_model() walks the pool and returns the first model under 75%
of its weekly free-tier limit.  Usage persists to data/token_usage.json
so it survives process restarts.
"""
import json
import threading
from datetime import date
from pathlib import Path

from src.config import DATA_DIR

_lock = threading.Lock()
_USAGE_FILE = DATA_DIR / "token_usage.json"
_ROTATION_THRESHOLD = 0.75

# Rotation order: Gemma → Nemotron Ultra → Nemotron Lightning.
# weekly_limit values are the OpenRouter free-tier weekly caps in tokens.
PUBLIC_MODEL_POOL: list[dict] = [
    {"id": "google/gemma-4-31b-it:free",             "weekly_limit": 613_000_000},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free",  "weekly_limit": 3_460_000_000_000},
    {"id": "nvidia/nemotron-3.5-lightning:free",      "weekly_limit": 1_200_000_000_000},
]


def _iso_week() -> str:
    iso = date.today().isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _load() -> dict:
    try:
        return json.loads(_USAGE_FILE.read_text()) if _USAGE_FILE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    _USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_FILE.write_text(json.dumps(data, indent=2))


def add_tokens(model_id: str, count: int) -> None:
    """Record `count` tokens used for `model_id` in the current ISO week."""
    week = _iso_week()
    with _lock:
        data = _load()
        entry = data.setdefault(model_id, {"week": week, "tokens": 0})
        if entry.get("week") != week:
            entry["week"] = week
            entry["tokens"] = 0
        entry["tokens"] += count
        _save(data)


def get_weekly_usage(model_id: str) -> int:
    """Return tokens used for `model_id` this ISO week (0 if the week rolled over)."""
    week = _iso_week()
    with _lock:
        data = _load()
    entry = data.get(model_id, {})
    return entry.get("tokens", 0) if entry.get("week") == week else 0


def is_over_threshold(model_id: str, weekly_limit: int) -> bool:
    return get_weekly_usage(model_id) >= weekly_limit * _ROTATION_THRESHOLD


def get_active_model() -> str:
    """Return the first pool model under the 75% weekly threshold.

    Falls back to the last model when all are saturated (graceful degradation).
    """
    for model in PUBLIC_MODEL_POOL:
        if not is_over_threshold(model["id"], model["weekly_limit"]):
            return model["id"]
    return PUBLIC_MODEL_POOL[-1]["id"]
