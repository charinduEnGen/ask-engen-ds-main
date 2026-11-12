"""Concurrency gate for the pipeline's two blocking calls: call_llm (network,
provider-side) and execute_sql_query (CPU/IO, in-process SQLite). Both are
plain sync functions, so every call site must run them off the event loop via
asyncio.to_thread — otherwise one slow LLM round-trip stalls every other
concurrent request this worker is serving.

The semaphore additionally caps how many LLM calls run at once: Gemini
Enterprise has its own concurrency/rate ceiling, and without a local cap,
10 concurrent users each firing a 2-call pipeline (agent 1 + agent 2) could
put 20 requests in flight against the provider at once. _llm_semaphore is
sized for 10 concurrent users comfortably (5 in-flight LLM calls, the rest
queued behind asyncio.wait_for's 15s acquire timeout).

The actual call_llm()/execute_sql_query() call sites live in
src.services.analytics.steps. This module is imported by both:
steps.py uses call_llm_guarded/run_pandas_exec at each call site, and
pipeline.py imports _llm_semaphore so the gate is visible at the
orchestrator layer too, without steps.py and pipeline.py importing
each other in a cycle.
"""

import asyncio
from typing import Callable, TypeVar

from fastapi import HTTPException

from src.config import settings

T = TypeVar("T")

_LLM_SEMAPHORE_LIMIT = 5
_ACQUIRE_TIMEOUT_SECONDS = 15.0
_CALL_TIMEOUT_SECONDS = 25.0

_llm_semaphore = asyncio.Semaphore(_LLM_SEMAPHORE_LIMIT)

_in_flight = 0


def get_call_timeout(mode: str) -> float:
    """Return the appropriate LLM call timeout for the given mode.

    Public mode uses OpenRouter free-tier models which are significantly
    slower than Gemini/Anthropic; they need a longer gate timeout.
    """
    from src.core.llm import store as llm_store

    if llm_store.get_mode(mode) == "public":
        return settings.public_llm_call_timeout_seconds
    return _CALL_TIMEOUT_SECONDS


async def call_llm_guarded(func: Callable[..., T], *args, call_timeout: float = _CALL_TIMEOUT_SECONDS, **kwargs) -> T:
    global _in_flight
    was_saturated = _llm_semaphore.locked()
    if was_saturated:
        print(f"[LLM_GATE] all {_LLM_SEMAPHORE_LIMIT} slots busy — waiting up to {_ACQUIRE_TIMEOUT_SECONDS}s for one to free up")
    try:
        await asyncio.wait_for(_llm_semaphore.acquire(), timeout=_ACQUIRE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        print(f"[LLM_GATE] BUSY — no free slot after {_ACQUIRE_TIMEOUT_SECONDS}s ({_in_flight}/{_LLM_SEMAPHORE_LIMIT} in flight) -> 503 Server busy")
        raise HTTPException(status_code=503, detail={"error": "Server busy", "retry_after": 20}) from exc

    _in_flight += 1
    print(f"[LLM_GATE] acquired slot — {_in_flight}/{_LLM_SEMAPHORE_LIMIT} in flight")
    try:
        return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=call_timeout)
    except asyncio.TimeoutError:
        print(f"[LLM_GATE] call itself timed out after {call_timeout}s")
        raise
    finally:
        _in_flight -= 1
        _llm_semaphore.release()
        print(f"[LLM_GATE] released slot — {_in_flight}/{_LLM_SEMAPHORE_LIMIT} in flight")


async def run_pandas_exec(func: Callable[..., T], *args, **kwargs) -> T:
    """Offloads a sync execute_sql_query(...)-shaped call to a worker thread."""
    print(f"[LLM_GATE] offloading {func.__name__} to a worker thread")
    return await asyncio.to_thread(func, *args, **kwargs)
