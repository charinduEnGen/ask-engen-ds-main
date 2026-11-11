# ask-engen-ds (Analytics Engine)

Generic multi-project analytics engine. A FastAPI service that lets a user
upload Excel file(s) for a project (each sheet becomes a table), auto-discovers
a schema for review, persists the confirmed data into a per-project SQLite
database, and then answers natural-language questions about it with text /
table / chart answers, backed by an LLM-driven SQL query pipeline.

## Overview

Per-project flow:

1. **`POST /datasets/ingest`** — the backend forwards uploaded Excel file(s)
   here. Each sheet becomes a candidate table; only its head is read and sent
   to the **`schema_relation_builder_agent`**, which proposes a column schema
   per table plus cross-table relationships, returned for the user to review.
2. **`POST /datasets/confirm`** — once the user accepts the (optionally
   edited) schema, the full sheet data is loaded into
   `data/<project>/<project>.db` (one SQLite file per project), and
   `schema.json` / `data_context.json` / `patterns.json` are written per
   table under `data/<project>/<table>/`. See
   [src/routes/datasets.py](src/routes/datasets.py).
3. **Query endpoints** are then backed by a modular pipeline:
   - **Agent 1 — `data_processing_agent`**: given the user's question, the
     project's schema (tables, columns, relationships), and conversation
     history, decides whether the question is answerable, resolves
     pronoun/follow-up references, and writes a single read-only SQL SELECT
     against the project's SQLite database. Retries on execution failure up
     to `MAX_CODE_GEN_RETRIES`.
   - **Response shaping**, based on result row count:
     - small (`<= SMALL_RESULT_THRESHOLD` rows): **Agent 2 —
       `final_answer_enrichment_agent`** turns the result into narrative text
       + table/chart + follow-up questions.
     - everything else (`> SMALL_RESULT_THRESHOLD` rows): a lighter follow-ups-only
       LLM call plus the full table sent in one shot (no narrative) — the client
       pages through the already-delivered rows locally, `DEFAULT_PAGE_SIZE` rows
       at a time.
   - **Session memory**: conversation history and mask maps are persisted per
     `(user_id, session_id, project)` — carried as `mode` in the request body —
     so follow-up turns can resolve references and reuse prior context.
     Read/writes are deferred to background tasks so they never add to
     response latency.

Sensitive columns are pseudonymized before being sent to any LLM prompt and
demasked before the response reaches the caller — see
[src/utils/masking.py](src/utils/masking.py).

## Prerequisites

- Python 3.12
- An API key for the configured LLM provider (Anthropic or Gemini).
- No external database — each project's data lives in its own SQLite file
  under `data/<project>/<project>.db`, created by `POST /datasets/confirm`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in the values below
```

### Environment variables

| Variable                                                          | Description                                                                                                                                    | Default                                     |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `LLM_PROVIDER`                                                    | `anthropic` \| `gemini`                                                                                                                        | `gemini`                                    |
| `ANTHROPIC_API_KEY`                                               | Required when `LLM_PROVIDER=anthropic`                                                                                                         | —                                           |
| `ANTHROPIC_MODEL`                                                 | Model id for Anthropic                                                                                                                         | `claude-opus-5`                             |
| `GEMINI_API_KEY`                                                  | Required when `LLM_PROVIDER=gemini`                                                                                                            | —                                           |
| `GEMINI_MODEL_DATA_PROCESSING`                                    | Model for agent 1 (carries the full schema/data_context/memory_context prompt)                                                                 | `gemini-3.6-flash`                          |
| `GEMINI_MODEL_ENRICHMENT`                                         | Model for agent 2 (small result preview / schema only, cheaper is fine)                                                                        | `gemini-3.5-flash-lite`                     |
| `LLM_MAX_TOKENS` / `LLM_MAX_OUTPUT_TOKENS`                        | Output token cap per LLM call                                                                                                                  | `4096` / `8192`                             |
| `LLM_MAX_INPUT_TOKENS`                                            | Pre-flight input token budget check                                                                                                            | `1_000_000`                                 |
| `GEMINI_CONTEXT_CACHE_ENABLED`                                    | Enable explicit Gemini context caching for agent 1's static context (see [Gemini context caching](#gemini-context-caching))                    | `true`                                      |
| `GEMINI_CACHE_TTL_SECONDS`                                        | TTL (seconds) for the Gemini `CachedContent`; set just under 24h so it naturally expires and rebuilds before the next day's data context lands | `82800` (23h)                               |
| `SMALL_RESULT_THRESHOLD`                                          | Row count boundary: enrichment agent runs at/under this                                                                                        | `20`                                        |
| `DEFAULT_PAGE_SIZE`                                               | Table pagination page size                                                                                                                     | `5`                                         |
| `MAX_CODE_GEN_RETRIES`                                            | Agent 1 retries on SQL execution failure                                                                                                       | `3`                                         |
| `DEFAULT_CURRENCY`                                                | Currency label used in responses                                                                                                               | `LKR`                                       |
| `SCHEMA_SAMPLE_ROWS`                                              | Rows of each uploaded sheet shown to the schema-discovery agent                                                                                | `5`                                         |

See [.env.example](.env.example) for the full list and
[src/config.py](src/config.py) for defaults/behavior.

## Running

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 5005 --reload
```

There's nothing to preload at startup — a project's SQLite engine and schema
files are created on its first `POST /datasets/confirm` and read lazily on
every query after that (see [src/db/engine.py](src/db/engine.py),
[src/utils/kb_loader.py](src/utils/kb_loader.py)).

- Health check: `GET /health`
- Dataset endpoints:
  - `POST /datasets/ingest` — upload Excel file(s) for a project, get back a
    proposed schema for review.
  - `POST /datasets/confirm` — persist the (optionally edited) schema and
    full data into that project's SQLite database.
- Query endpoint:
  - `POST /query/{project}`

Query requests use the `UserRequest` body — see
[src/schemas/user_request.py](src/schemas/user_request.py) and
[payloads/payload_v1.md](payloads/payload_v1.md) for example payloads.
`mode` in that body carries the project name end to end from the backend.

## Data & runtime state (git-ignored)

- `data/<project>/<project>.db` — that project's SQLite database, created on
  confirm.
- `data/<project>/<table>/{schema,data_context,patterns}.json` — per-table
  schema, context, and reusable query patterns, written on confirm and read
  on every query turn.
- `data/<project>/_staging/<upload_id>/` — raw uploaded files, held between
  ingest and confirm, deleted once confirm runs.
- `data/sessions/` — per-session conversation memory JSON files.
- `logs/` — per-request stage timing logs.
- `costs/` — LLM call cost logs.

None of these are committed; the app (re)creates what it needs at runtime.

## Gemini context caching

When `LLM_PROVIDER=gemini`, agent 1's static prompt block — the column
schema, data context (date range, available months, row count), analytics
patterns, and the task's literal instructions/output contract — is kept in
an **explicit Gemini `CachedContent`** object rather than re-sent (and
re-billed) on every call. Only the volatile per-request suffix (the user's
question, resolved merchant filter, conversation memory) is sent afresh each
turn.

### How it works

1. On the first agent 1 call (or after a content change / cache expiry),
   `src/utils/gemini_cache.py::get_or_create_cache` calls
   `CachedContent.create` with the system instruction + static context and
   registers the result in an in-process registry keyed by
   `mode:model:sha256(system_instruction + static_context)[:16]`.
2. Subsequent calls within the same process get the live `CachedContent`
   object from the registry — no round-trip to create a new one — and
   `genai.GenerativeModel.from_cached_content` wraps it so the provider
   serves the cached tokens at the discounted `cache_read` rate.
3. The registry entry is refreshed proactively `GEMINI_CACHE_TTL_SECONDS - 24hr`
   before it expires, so no in-flight request ever races an already-expired
   cache.

### Cache invalidation

The cache key is a hash of the _exact content_, so any edit to a KB file or
a new day's data context (different `CURRENT_DATE` / `AVAILABLE_MONTHS` /
`TOTAL_ROWS`) produces a different key, causing an automatic miss and fresh
cache creation — no manual invalidation required. The old `CachedContent`
simply ages out on Gemini's side once its TTL elapses.

### Fallback behaviour

If `CachedContent.create` fails for any reason (content below Gemini's
minimum cacheable size, a transient API error, the feature not available for
the selected model), `get_or_create_cache` returns `None` and the call
degrades to sending the static context inline — identical to pre-caching
behaviour — with no user-visible impact.

### Cost impact

Gemini bills cached tokens at a heavily discounted `cache_read` rate
compared to fresh input tokens (see `MODEL_PRICING` in
[src/config.py](src/config.py)):

| Model                   | Input ($/1M) | Cache read ($/1M) | Output ($/1M) |
| ----------------------- | ------------ | ----------------- | ------------- |
| `gemini-3.6-flash`      | 1.50         | 0.15 (10×)        | 7.50          |
| `gemini-3.5-flash-lite` | 0.30         | 0.03 (10×)        | 2.50          |

Every call's `cached_tokens` count is logged to `costs/cost_YYYY-MM-DD.jsonl`
alongside the matching `cache_cost_usd` breakdown so you can verify the
savings over time.

### Disabling caching

Set `GEMINI_CONTEXT_CACHE_ENABLED=false` (or switch to
`LLM_PROVIDER=anthropic`) to bypass the cache entirely. The Anthropic path
never uses explicit caching — it receives the full static context inline
every call.

## Prompt & knowledge configuration

- [src/agents/agents.yml](src/agents/agents.yml) — role/goal/backstory for
  `data_processing_agent`, `final_answer_enrichment_agent`, and
  `schema_relation_builder_agent`.
- [src/tasks/tasks.yml](src/tasks/tasks.yml) — per-task instructions and
  output contracts (schema discovery, data analysis, answer enrichment,
  follow-ups).
- `data/<project>/<table>/{schema,data_context,patterns}.json` — per-project,
  per-table column schema, data context, and analytics pattern references
  injected into agent 1's prompt (see [src/utils/kb_loader.py](src/utils/kb_loader.py)).

Editing these files changes agent behavior without touching application
code.
