"""Per-request time tracker, shared across every project. Appends one entry per
Q&A to logs/timing_YYYY-MM-DD.jsonl with a wall-clock breakdown per pipeline
stage, tagged by user/session/mode so it can be sliced later — see
src.routes.dashboard. See src.utils.cost_utils' module docstring for why
JSONL over a read-modify-write YAML document.
"""

import asyncio
import json
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import yaml

from src.config import LOGS_DIR

# Fixed +5:30 offset, no DST — safe to hardcode rather than pull in a tz
# database dependency just for this one console-facing conversion.
_IST = timezone(timedelta(hours=5, minutes=30))


def log_timestamp() -> str:
    """Current IST time formatted for a console-log prefix — shared by
    console_log below and main.py's [STARTUP]/[SHUTDOWN] prints, so every
    stdout log line uses one consistent timestamp format. IST purely for
    terminal readability; the persisted JSONL/YAML logs (log_timing,
    log_cost, log_agent1_failure below) intentionally stay on UTC
    timestamps, so don't reuse this for anything that gets written to
    disk."""
    return datetime.now(_IST).strftime("%Y-%m-%d %H:%M:%S IST")


def console_log(mode: str, topic: str, message: str) -> None:
    """Console print with a consistent [TIMESTAMP][MODE][TOPIC] prefix, so
    different projects' requests interleaved on stdout (e.g. two Postman
    calls in flight at once) stay easy to follow, grep by topic, and
    order/correlate by time."""
    print(f"[{log_timestamp()}][{mode}][{topic}] {message}")


class StageTimer:
    def __init__(self) -> None:
        self._stages: dict[str, float] = {}

    @contextmanager
    def track(self, stage_name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[stage_name] = round(time.perf_counter() - start, 4)

    @property
    def stages(self) -> dict[str, float]:
        return dict(self._stages)

    @property
    def total_seconds(self) -> float:
        return round(sum(self._stages.values()), 4)


# Separate lock instance from src.utils.cost_utils._log_lock — timing and
# cost entries are written to different files/directories, so there's no
# reason to serialize one behind the other.
_log_lock = asyncio.Lock()


def _append_jsonl_line(path, line: str) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def log_timing(
    user_id: str,
    session_id: str,
    answer_id: str,
    query: str,
    stages: dict[str, float],
    total_seconds: float,
    mode: str = "unknown",
) -> None:
    path = LOGS_DIR / f"timing_{date.today()}.jsonl"

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "answer_id": answer_id,
        "mode": mode,
        "query": query,
        "stages_seconds": stages,
        "total_seconds": total_seconds,
    }

    async with _log_lock:
        await asyncio.to_thread(_append_jsonl_line, path, json.dumps(entry, ensure_ascii=False))
    console_log(mode, "TIMING", f"answer_id={answer_id} total_seconds={total_seconds} -> {path.name}")


def log_agent1_failure(
    user_id: str,
    session_id: str,
    answer_id: str,
    query: str,
    mode: str,
    scope_category: str,
    out_of_scope_reason: str,
    failed_attempts: list[dict],
) -> None:
    """Persists the code + error from every failed attempt when a turn
    exhausts src.orchestrator.steps.run_data_processing_agent's retries
    (scope_category == "system_error") — the only place this detail survives
    past the request; console_log (see above) is stdout-only, and log_timing
    above only ever records how long the stage took, not why it failed.
    Written to its own per-day file rather than folded into log_timing's so
    the common case (successful turns) doesn't carry this extra weight."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOGS_DIR / f"{day}_agent1_failures.yml"

    entries = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            entries = yaml.safe_load(f) or []

    entries.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "session_id": session_id,
            "answer_id": answer_id,
            "mode": mode,
            "query": query,
            "scope_category": scope_category,
            "out_of_scope_reason": out_of_scope_reason,
            "attempts": failed_attempts,
        }
    )

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(entries, f, sort_keys=False, allow_unicode=True)
        