"""Single-layer conversation memory, shared across every project.

This is the definitive location; src/utils/session_memory.py is a
backward-compat re-export.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import SESSIONS_DIR, settings

_SAFE_RE = re.compile(r"[^A-Za-z0-9_\-]")
_CHAT_EXPIRY = timedelta(hours=24)
_MAX_CONTEXT_TOKEN_RATIO = 0.5

_session_locks: dict[str, asyncio.Lock] = {}


def _get_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
        print(f"[SESSION_MEMORY] created lock for session key={session_id!r} ({len(_session_locks)} session lock(s) tracked)")
    return _session_locks[session_id]


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _session_file(user_id: str, session_id: str, mode: str) -> Path:
    safe_user = _SAFE_RE.sub("_", user_id)
    safe_session = _SAFE_RE.sub("_", session_id)
    safe_mode = _SAFE_RE.sub("_", mode)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"{safe_user}__{safe_session}__{safe_mode}.json"


def _load_turns(user_id: str, session_id: str, mode: str) -> list[dict]:
    path = _session_file(user_id, session_id, mode)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_history(user_id: str, session_id: str, mode: str) -> list[dict]:
    return _load_turns(user_id, session_id, mode)


def count_turns(user_id: str, session_id: str, mode: str) -> int:
    return len(_load_turns(user_id, session_id, mode))


def is_context_window_full(user_id: str, session_id: str, mode: str) -> bool:
    turns = _load_turns(user_id, session_id, mode)
    if not turns:
        return False
    estimated_tokens = _estimate_tokens(format_history_for_prompt(turns))
    threshold = settings.llm_max_input_tokens * _MAX_CONTEXT_TOKEN_RATIO
    return estimated_tokens >= threshold


def is_chat_expired(user_id: str, session_id: str, mode: str) -> bool:
    turns = _load_turns(user_id, session_id, mode)
    if not turns:
        return False
    first_turn_at = datetime.fromisoformat(turns[0]["timestamp"])
    return datetime.now(timezone.utc) - first_turn_at > _CHAT_EXPIRY


async def append_turn(
    user_id: str,
    session_id: str,
    mode: str,
    answer_id: str,
    query: str,
    answer: dict[str, Any],
    response_type: str,
    mask_map: Optional[dict] = None,
    pending_disambiguation: Optional[dict] = None,
) -> None:
    path = _session_file(user_id, session_id, mode)
    lock_key = f"{user_id}:{session_id}:{mode}"
    lock = _get_lock(lock_key)
    if lock.locked():
        print(f"[SESSION_MEMORY] session key={lock_key!r} busy with another write — waiting for lock (answer_id={answer_id})")
    async with lock:
        turns: list[dict] = []
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                turns = json.load(f)

        turn = {
            "answer_id": answer_id,
            "query": query,
            "answer": answer,
            "response_type": response_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if mask_map:
            turn["mask_map"] = mask_map
        if pending_disambiguation:
            turn["pending_disambiguation"] = pending_disambiguation
        turns.append(turn)

        await asyncio.to_thread(path.write_text, json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SESSION_MEMORY] appended turn answer_id={answer_id} for session key={lock_key!r} -> {len(turns)} turn(s) total")


def get_mask_map_for_answer(user_id: str, session_id: str, mode: str, answer_id: str) -> Optional[dict]:
    if not answer_id:
        return None
    for turn in _load_turns(user_id, session_id, mode):
        if turn.get("answer_id") == answer_id:
            return turn.get("mask_map")
    return None


def get_pending_disambiguation(user_id: str, session_id: str, mode: str, answer_id: str) -> Optional[dict]:
    if not answer_id:
        return None
    for turn in _load_turns(user_id, session_id, mode):
        if turn.get("answer_id") == answer_id:
            return turn.get("pending_disambiguation")
    return None


def format_history_for_prompt(turns: list[dict]) -> str:
    if not turns:
        return "No prior conversation for this session."
    lines = []
    for turn in turns:
        lines.append(f"User: {turn['query']}")
        lines.append(f"Assistant ({turn['response_type']}): {json.dumps(turn['answer'], ensure_ascii=False)}")
    return "\n".join(lines)


def purge_expired_sessions() -> int:
    """Deletes session files whose first turn is older than _CHAT_EXPIRY.
    Returns the number of files deleted."""
    deleted = 0
    if not SESSIONS_DIR.exists():
        return deleted
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                turns = json.load(f)
            if not turns:
                continue
            first_turn_at = datetime.fromisoformat(turns[0]["timestamp"])
            if datetime.now(timezone.utc) - first_turn_at > _CHAT_EXPIRY:
                path.unlink()
                deleted += 1
        except Exception:  # noqa: BLE001
            pass
    return deleted
