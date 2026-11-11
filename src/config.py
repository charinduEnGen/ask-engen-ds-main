from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
COSTS_DIR = PROJECT_ROOT / "costs"
SESSIONS_DIR = DATA_DIR / "sessions"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1M tokens. `cache_read` is what a cache-hit token on the
    *input* side costs (Gemini explicit context caching — see
    src.utils.gemini_cache) — always cheaper than a fresh input token, and
    separate from `input`, which only applies to the non-cached portion of
    the prompt. 0.0 for models/providers with no caching discount."""

    input: float
    output: float
    cache_read: float = 0.0


# Keyed by the exact model string returned in LLMResult.model (i.e.
# settings.gemini_model_data_processing / gemini_model_enrichment /
# anthropic_model) — src.utils.cost_utils.compute_cost looks a call's
# `model` up here directly. Update this alongside the settings above
# whenever a model is swapped or a provider changes its rate card.
MODEL_PRICING: dict[str, ModelPricing] = {
    # Gemini — rates per Google's published Standard Pay-As-You-Go pricing.
    "gemini-3.6-flash": ModelPricing(input=1.50, output=7.50, cache_read=0.15),
    "gemini-3.5-flash-lite": ModelPricing(input=0.30, output=2.50, cache_read=0.03),
    # Anthropic — no explicit cache_read rate wired up yet (_call_anthropic
    # doesn't use prompt caching), so cache_read stays at the 0.0 default.
    "claude-opus-5": ModelPricing(input=5.00, output=25.00),
    "claude-sonnet-5": ModelPricing(input=3.00, output=15.00),
    "claude-haiku-4-5": ModelPricing(input=1.00, output=5.00),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM provider — switch with LLM_PROVIDER env var, no code change needed.
    llm_provider: Literal["anthropic", "gemini"] = "gemini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-5"
    gemini_api_key: Optional[str] = None
    # Two separate models: agent 1 (data_processing_agent) carries the heavy
    # prompt (schema + data_context + memory_context + analytics_patterns)
    # and needs the stronger model; agent 2 (final_answer_enrichment_agent,
    # used by both answer_enrichment_task and followups_task) only ever sees
    # a small result preview or just the schema, so a lighter/cheaper model
    # is enough. See src.utils.llm_client.call_llm's `gemini_model` param and
    # the call sites in src.orchestrator.steps.
    gemini_model_data_processing: str = "gemini-3.5-flash-lite"
    gemini_model_enrichment: str = "gemini-3.5-flash-lite"
    # Informational only — the Active/Inactive badge on /dashboard (see
    # src.routes.dashboard) flips to Inactive once today's date passes this;
    # nothing in the pipeline itself reads these to gate a call. None means
    # "no known deprecation date" -> always shown Active.
    gemini_model_data_processing_deprecation_date: Optional[date] = None
    gemini_model_enrichment_deprecation_date: Optional[date] = None
    # Explicit Gemini context caching (see src.utils.gemini_cache) for agent
    # 1's static context — schema + data_context + analytics_patterns, plus
    # the task's literal instructions/output contract — none of which vary
    # per request, only when a project's schema/data_context files change.
    # Off switch for local dev / providers other than gemini, where it's a
    # no-op anyway.
    gemini_context_cache_enabled: bool = True
    # How long Gemini keeps the cached content alive before it must be
    # recreated. The cache key (src.utils.gemini_cache._cache_key) already
    # hashes mode+model+content, so a longer TTL is a pure cost/latency win,
    # never a correctness risk: the moment a project's schema/data_context
    # changes (re-confirm with new data), the hash changes, the old key just
    # misses, and a fresh cache is created on the next request — no manual
    # invalidation needed. Set to just under 24h (not a full day) so the
    # cache naturally expires and gets rebuilt slightly ahead of the next
    # day's data_context landing, rather than racing it. Gemini itself
    # imposes no min/max bound on ttl (defaults to 1h if unset) — see
    # https://ai.google.dev/api/caching.
    gemini_cache_ttl_seconds: int = 82_800  # 23h
    # Output cap — passed as generation_config.max_output_tokens (Gemini) /
    # max_tokens (Anthropic). Keep this small; it bounds a single response,
    # not the conversation. A too-large value here gets rejected outright by
    # the provider as an invalid argument.
    llm_max_output_tokens: int = 8192
    # Input cap — rough pre-flight budget for system+user prompt size
    # (schema + data_context + memory_context + analytics_patterns), checked
    # in src.utils.llm_client before the call goes out. gemini_model_data_processing's
    # own context window is ~1M tokens; this just fails fast with
    # TokenLimitExceededError instead of spending a round-trip on a prompt
    # the provider would reject anyway.
    llm_max_input_tokens: int = 1_000_000

    # Response-shaping threshold (row count of the SQL result). <= this
    # many rows runs the enrichment agent (narrative + table/chart); above
    # it, every size — 21 rows or 20,000 — gets the same follow-ups-only
    # agent + full table (client-paged) treatment. See query_orchestrator.
    small_result_threshold: int = 20
    default_page_size: int = 5

    # Agent 1 retries when its generated SQL fails to execute.
    max_code_gen_retries: int = 3

    default_currency: str = "LKR"

    # Dataset-discovery agent: how many rows of each uploaded sheet to show
    # it when inferring column types/descriptions/relationships. See
    # src.routes.datasets.ingest.
    schema_sample_rows: int = 5
    # Multer on the backend already caps this at 150MB/file; this is the
    # ds-side mirror so a direct call to this service is bounded too.
    max_upload_file_size_mb: int = 150
    # Schema discovery is a one-off call with a much heavier prompt (every
    # uploaded table's full header + sample rows at once) than a single query
    # turn, so it gets its own longer timeout instead of the ~25s default
    # tuned for live chat latency (see src.orchestrator.llm_gate.call_llm_guarded).
    schema_discovery_timeout_seconds: float = 90.0


settings = Settings()
