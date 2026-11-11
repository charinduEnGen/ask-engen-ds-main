"""Per-section token/cost tracker, shared across every project. Appends one entry
per LLM call to costs/cost_YYYY-MM-DD.jsonl (input/output/cached tokens + an
estimated USD cost breakdown), tagged by user/session/answer/mode and
pipeline section (e.g. "agent1_query_build_attempt_1", "agent2_enrichment",
"followups_only"). `mode` defaults to "unknown" so entries written before
this field existed (or by any future caller that forgets it) still parse
the same shape everywhere they're read back — see src.routes.dashboard.

Append-only JSONL (one JSON object per line) rather than a single
read-modify-write YAML/JSON document — under concurrent requests, a
read-the-whole-file-then-rewrite-it pattern is a lost-update race even with
a lock serializing writers (still fine here since _log_lock does serialize
them), but JSONL also means a crash mid-write can only ever corrupt the last
line, never the whole day's log, and readers (src.routes.dashboard) can
stream it instead of loading one giant structure.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone

from src.config import MODEL_PRICING, COSTS_DIR
from src.utils.logging_utils import console_log


@dataclass
class CostBreakdown:
    """Result of compute_cost — kept split out (rather than just the total)
    so log_cost can persist input/output/cache cost separately and
    estimated_cost_usd's provenance is auditable from the yaml alone."""

    input_cost_usd: float
    output_cost_usd: float
    cache_cost_usd: float

    @property
    def total_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd + self.cache_cost_usd


def compute_cost(
    model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0
) -> CostBreakdown:
    """input_tokens is the *total* prompt size (Gemini's prompt_token_count
    already includes any cached portion), so the non-cached slice billed at
    the full input rate is input_tokens - cached_tokens; cached_tokens itself
    is billed at the model's (cheaper) cache_read rate. Unknown model ->
    zeroed breakdown rather than raising, so a rate-card gap degrades to
    "cost unknown", not a broken pipeline."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return CostBreakdown(0.0, 0.0, 0.0)

    cached_tokens = min(cached_tokens, input_tokens)
    billable_input_tokens = input_tokens - cached_tokens

    return CostBreakdown(
        input_cost_usd=billable_input_tokens / 1_000_000 * pricing.input,
        output_cost_usd=output_tokens / 1_000_000 * pricing.output,
        cache_cost_usd=cached_tokens / 1_000_000 * pricing.cache_read,
    )


# Serializes every append across concurrent requests in this worker process
# — JSONL appends are individually safe from `open(..., "a")`'s OS-level
# atomicity for a single write() of a short line, but the lock still avoids
# interleaving two entries' bytes under a hostile write size / platform, and
# keeps the offloaded-to-a-thread file open+write+close as one unit.
_log_lock = asyncio.Lock()


def _append_jsonl_line(path, line: str) -> None:
    COSTS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def log_cost(
    user_id: str,
    session_id: str,
    answer_id: str,
    section: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    mode: str = "unknown",
) -> None:
    path = COSTS_DIR / f"cost_{date.today()}.jsonl"
    cost = compute_cost(model, input_tokens, output_tokens, cached_tokens)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "answer_id": answer_id,
        "mode": mode,
        "section": section,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "input_cost_usd": round(cost.input_cost_usd, 6),
        "output_cost_usd": round(cost.output_cost_usd, 6),
        "cache_cost_usd": round(cost.cache_cost_usd, 6),
        "estimated_cost_usd": round(cost.total_usd, 6),
    }

    async with _log_lock:
        await asyncio.to_thread(_append_jsonl_line, path, json.dumps(entry, ensure_ascii=False))
    console_log(
        mode,
        "COST",
        f"answer_id={answer_id} section={section} model={model} "
        f"tokens(in/out/cached)={input_tokens}/{output_tokens}/{cached_tokens} "
        f"cost=${entry['estimated_cost_usd']:.6f} -> {path.name}",
    )
