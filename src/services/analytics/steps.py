"""The LLM-driven steps of the analytics pipeline.

Moved from src/orchestrator/steps.py; that file is now a backward-compat
re-export.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd
from sqlalchemy.engine import Engine

from src.config import settings
from src.core.llm.gate import call_llm_guarded, get_call_timeout, run_pandas_exec
from src.services.analytics.tools import response_builder
from src.services.analytics.tools.sql_query_tool import execute_sql_query
from src.utils import kb_loader, masking, prompt_loader
from src.utils.cost_utils import log_cost
from src.utils.external_db_loader import fetch_url_content, get_source_by_key, load_external_db
from src.utils.glossary_loader import render_glossary_block
from src.core.llm.client import call_llm
from src.utils.logging_utils import console_log

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def _extract_json(text: str) -> dict:
    return json.loads(_strip_fences(text))


@dataclass
class Agent1Outcome:
    kind: Literal["result", "out_of_scope"]
    result_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    query_used: str = ""
    out_of_scope_reason: Optional[str] = None
    scope_category: Optional[str] = None
    failed_attempts: list[dict] = field(default_factory=list)


_GENERIC_OUT_OF_SCOPE_REASON = "repeated malformed responses or failing code across every retry attempt"


async def run_data_processing_agent(
    *,
    mode: str,
    query: str,
    schema: dict,
    data_context: str,
    analytics_patterns: list[dict],
    memory_context: str,
    engine: Engine | None,
    user_id: str,
    session_id: str,
    answer_id: str,
    reply_to_mask_key: masking.MaskKey | None = None,
) -> Agent1Outcome:
    system_prompt = prompt_loader.build_agent_system_prompt("data_processing_agent")

    business_glossary = render_glossary_block(mode)
    static_context = prompt_loader.render_task_static_context(
        "data_analysis_task",
        mode=mode,
        data_context=data_context,
        schema=json.dumps(schema, ensure_ascii=False),
        analytics_patterns=json.dumps(analytics_patterns, ensure_ascii=False),
        business_glossary=business_glossary,
    )
    volatile_prompt = prompt_loader.render_task_volatile_prompt(
        "data_analysis_task",
        query=query,
        memory_context=memory_context,
    )
    gemini_static_context = static_context if settings.llm_provider == "gemini" and static_context else None
    base_prompt = (
        volatile_prompt
        if gemini_static_context
        else (f"{static_context}\n\n{volatile_prompt}" if static_context else volatile_prompt)
    )

    console_log(mode, "AGENT1_MEMORY", f"memory_context passed to prompt:\n{memory_context}")

    user_prompt = base_prompt
    last_error = "unknown error"
    failed_attempts: list[dict] = []
    for attempt in range(1, settings.max_code_gen_retries + 1):
        result = await call_llm_guarded(
            call_llm,
            mode,
            system_prompt,
            user_prompt,
            call_timeout=get_call_timeout(mode),
            gemini_model=settings.gemini_model_data_processing,
            gemini_static_context=gemini_static_context,
        )
        await log_cost(
            user_id,
            session_id,
            answer_id,
            f"agent1_query_build_attempt_{attempt}",
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.cached_tokens,
            mode,
        )

        raw_text = _strip_fences(result.text)
        console_log(
            mode,
            "AGENT1_QUERY",
            f"attempt {attempt}/{settings.max_code_gen_retries}: raw decision:\n{raw_text}",
        )

        try:
            decision = _extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = f"TOOL_ERROR: response was not valid JSON: {exc}"
            console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: {last_error}")
            failed_attempts.append(
                {"attempt": attempt, "failure": "invalid_json", "raw_text": raw_text[:2000], "error": last_error}
            )
            user_prompt = (
                base_prompt
                + f"\n\nYour previous response was not valid JSON:\n{raw_text}\n\n"
                "Return ONLY the JSON object described above, no markdown fences."
            )
            continue

        console_log(
            mode,
            "AGENT1_REASONING",
            f"attempt {attempt}: confidence={decision.get('confidence') or 'unknown'} "
            f"reason={(decision.get('reasoning') or '').strip()!r}",
        )

        if not decision.get("in_scope", True):
            reason = decision.get("out_of_scope_reason") or _GENERIC_OUT_OF_SCOPE_REASON
            category = (decision.get("scope_category") or "unsupported_data").strip()
            console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: out of scope (category={category}) -> {reason}")
            return Agent1Outcome(kind="out_of_scope", out_of_scope_reason=reason, scope_category=category)

        raw_sql = (decision.get("sql") or "").strip()
        if not raw_sql:
            last_error = "TOOL_ERROR: 'in_scope' was true but 'sql' was empty"
            console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: {last_error}")
            failed_attempts.append({"attempt": attempt, "failure": "empty_sql", "error": last_error})
            user_prompt = (
                base_prompt
                + f"\n\nYour previous response:\n{raw_text}\n\n{last_error}\n\n"
                "Return the JSON object again with a non-empty 'sql' field."
            )
            continue

        exec_sql = raw_sql
        if reply_to_mask_key is not None and not reply_to_mask_key.is_empty():
            exec_sql = masking.demask_json(raw_sql, reply_to_mask_key)
            if exec_sql != raw_sql:
                console_log(
                    mode,
                    "MASKING",
                    f"attempt {attempt}: demasked pseudonym(s) before execution:\n{exec_sql}",
                )

        # SQL-only mode: return SQL without executing (BE will execute it)
        if engine is None:
            console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: engine=None, returning sql without execution")
            return Agent1Outcome(kind="result", result_df=pd.DataFrame(), query_used=exec_sql)

        exec_outcome = await run_pandas_exec(execute_sql_query, exec_sql, engine)
        if isinstance(exec_outcome, pd.DataFrame):
            console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: execution succeeded -> {len(exec_outcome)} row(s)")
            return Agent1Outcome(kind="result", result_df=exec_outcome, query_used=exec_sql)

        last_error = exec_outcome
        console_log(mode, "AGENT1_QUERY", f"attempt {attempt}: execution failed -> {exec_outcome}")
        failed_attempts.append(
            {"attempt": attempt, "failure": "execution_error", "sql": raw_sql, "error": str(exec_outcome)}
        )
        user_prompt = (
            base_prompt
            + f"\n\nYour previous response:\n{raw_text}\n\nThe query failed:\n{exec_outcome}\n\n"
            "Return the JSON object again with in_scope true and a corrected 'sql' field — a single read-only SELECT statement."
        )

    console_log(mode, "AGENT1_QUERY", f"exhausted {settings.max_code_gen_retries} attempt(s), giving up -> {last_error}")
    return Agent1Outcome(
        kind="out_of_scope",
        out_of_scope_reason=_GENERIC_OUT_OF_SCOPE_REASON,
        scope_category="system_error",
        failed_attempts=failed_attempts,
    )


_FALLBACK_OUT_OF_SCOPE_TEXT = (
    "I wasn't able to process this request. Could you rephrase your "
    "question or narrow the scope?"
)


async def run_out_of_scope_response_agent(
    *,
    mode: str,
    query: str,
    scope_category: str,
    out_of_scope_reason: str,
    answer_id: str,
    user_id: str,
    session_id: str,
) -> str:
    system_prompt = prompt_loader.build_agent_system_prompt("final_answer_enrichment_agent")
    prompt = prompt_loader.render_task_prompt(
        "out_of_scope_response_task",
        mode=mode,
        query=query,
        scope_category=scope_category,
        out_of_scope_reason=out_of_scope_reason,
        data_freshness=kb_loader.data_freshness_note(mode),
    )

    result = await call_llm_guarded(call_llm, mode, system_prompt, prompt, call_timeout=get_call_timeout(mode), gemini_model=settings.gemini_model_enrichment)
    await log_cost(
        user_id,
        session_id,
        answer_id,
        "out_of_scope_response",
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        mode,
    )
    console_log(mode, "AGENT2_OUT_OF_SCOPE", f"category={scope_category} raw decision:\n{result.text}")

    try:
        decision = _extract_json(result.text)
    except (ValueError, json.JSONDecodeError):
        return _FALLBACK_OUT_OF_SCOPE_TEXT

    return decision.get("text") or _FALLBACK_OUT_OF_SCOPE_TEXT


async def run_enrichment_agent(
    *,
    mode: str,
    query: str,
    schema: dict,
    result_df: pd.DataFrame,
    answer_id: str,
    user_id: str,
    session_id: str,
) -> tuple[str | None, dict | None, dict | None, list[str] | None]:
    system_prompt = prompt_loader.build_agent_system_prompt("final_answer_enrichment_agent")
    prompt = prompt_loader.render_task_prompt(
        "answer_enrichment_task",
        query=query,
        schema=json.dumps(schema, ensure_ascii=False),
        row_count=str(len(result_df)),
        columns=json.dumps(list(result_df.columns)),
        result_preview=json.dumps(response_builder.dataframe_to_rows(result_df), ensure_ascii=False),
        data_freshness=kb_loader.data_freshness_note(mode),
    )

    result = await call_llm_guarded(call_llm, mode, system_prompt, prompt, call_timeout=get_call_timeout(mode), gemini_model=settings.gemini_model_enrichment)
    await log_cost(
        user_id,
        session_id,
        answer_id,
        "agent2_enrichment",
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        mode,
    )
    console_log(mode, "AGENT2_ENRICHMENT", f"raw decision:\n{result.text}")

    try:
        decision = _extract_json(result.text)
    except (ValueError, json.JSONDecodeError):
        decision = {"format": "text_table", "text": ""}

    follow_ups = list(decision.get("followups") or [])[:2] or None
    text_content = decision.get("text") or None
    fmt = decision.get("format", "text_table")

    if fmt == "text":
        return text_content, None, None, follow_ups

    table = response_builder.build_table_payload(
        result_df, page=1, page_size=settings.default_page_size, paginated=False
    )

    if fmt != "text_table_chart":
        return text_content, table, None, follow_ups

    chart_type = decision.get("chart_type")
    if chart_type not in {"pie_chart", "bar_chart", "line_chart"}:
        chart_type = "bar_chart"

    chart_cfg = decision.get("chart") or {}
    columns = list(result_df.columns)
    label_col = chart_cfg.get("category_column") if chart_cfg.get("category_column") in columns else columns[0]
    value_col = chart_cfg.get("value_column") if chart_cfg.get("value_column") in columns else columns[-1]
    unit = chart_cfg.get("unit") or settings.default_currency
    title = decision.get("title") or query

    chart = response_builder.build_chart_payload(chart_type, title, result_df, label_col, value_col, unit)
    return text_content, table, chart, follow_ups


@dataclass
class IntentionOutcome:
    intent: str
    source_key: Optional[str] = None
    reasoning: str = ""


_RAG_KEYWORDS = frozenset({
    "document", "documents", "report", "reports", "wiki", "article", "articles",
    "memo", "memos", "policy", "policies", "file says", "according to",
    "knowledge base", "summarize", "summary of", "what does the", "find in",
    "search document", "in the document", "from the document", "pdf", "docx",
    "internal doc", "internal docs", "uploaded doc", "uploaded file",
})


async def run_intention_classifier(
    *,
    mode: str,
    query: str,
    memory_context: str,
    answer_id: str,
    user_id: str,
    session_id: str,
) -> IntentionOutcome:
    from src.db import vector_store as _vs

    sources = load_external_db(mode)
    if sources:
        q_lower = query.lower()

        def _key_matches_in(key_name: str, text: str) -> bool:
            k = key_name.lower()
            if not k:
                return False
            if k in text:
                return True
            if k.replace("_", " ").replace("-", " ") in text:
                return True
            parts = [p for p in re.split(r"[_\-]", k) if len(p) > 2]
            return any(p in text for p in parts)

        def _key_matches(key_name: str) -> bool:
            return _key_matches_in(key_name, q_lower)

        # Also trigger when a prior turn's answer already referenced a source —
        # the user is following up on an external answer without repeating the
        # source name in their new question.
        mem_lower = memory_context.lower()
        memory_trigger = any(
            _key_matches_in(s.get("key_name", ""), mem_lower) for s in sources
        )

        # direct_trigger: user explicitly named the source or said "external" in THIS query.
        # memory_trigger alone should NOT bypass the LLM — it only helps the LLM decide.
        direct_trigger = (
            "external" in q_lower
            or any(_key_matches(s.get("key_name", "")) for s in sources)
        )
        explicit_trigger = direct_trigger or memory_trigger

        if explicit_trigger:
            # Fast-path: skip LLM only when user directly named the source in this query.
            # If memory_trigger is the only reason, let the LLM decide instead.
            if len(sources) == 1 and direct_trigger:
                matched_key = sources[0]["key_name"]
                console_log(mode, "INTENTION", f"intent=external source_key={matched_key} (single source, explicit trigger)")
                return IntentionOutcome(intent="external", source_key=matched_key, reasoning="explicit external trigger in query")

            external_sources_text = "\n".join(
                f"- key_name: {s.get('key_name')}, description: {s.get('description', '')}, link: {s.get('link', '')}"
                for s in sources
            )
            system_prompt = prompt_loader.build_agent_system_prompt("data_processing_agent")
            prompt = prompt_loader.render_task_prompt(
                "intention_classification_task",
                query=query,
                external_sources=external_sources_text,
                memory_context=memory_context,
            )
            try:
                result = await call_llm_guarded(
                    call_llm, mode, system_prompt, prompt,
                    call_timeout=get_call_timeout(mode),
                    gemini_model=settings.gemini_model_data_processing,
                )
                await log_cost(
                    user_id, session_id, answer_id, "intention_classifier",
                    result.model, result.input_tokens, result.output_tokens, result.cached_tokens, mode,
                )
                decision = _extract_json(result.text)
                intent = decision.get("intent", "internal")
                source_key = decision.get("source_key")
                reasoning = decision.get("reasoning", "")
                console_log(mode, "INTENTION", f"intent={intent} source_key={source_key} reason={reasoning!r}")
                if intent in ("external", "both") and source_key:
                    return IntentionOutcome(intent=intent, source_key=source_key, reasoning=reasoning)
            except Exception as exc:  # noqa: BLE001
                console_log(mode, "INTENTION", f"classifier failed ({exc}), falling through to rag/internal")

    kb_docs = _vs.list_documents(mode)
    if kb_docs:
        q_lower = query.lower()
        if any(kw in q_lower for kw in _RAG_KEYWORDS):
            console_log(mode, "INTENTION", f"intent=rag (keyword match, {len(kb_docs)} doc(s) indexed)")
            return IntentionOutcome(intent="rag", reasoning="query matches document-retrieval keywords and knowledge docs are present")

    console_log(mode, "INTENTION", "intent=internal")
    return IntentionOutcome(intent="internal", reasoning="default internal routing")


async def run_external_query_agent(
    *,
    mode: str,
    query: str,
    source_key: str,
    answer_id: str,
    user_id: str,
    session_id: str,
    internal_df: pd.DataFrame | None = None,
) -> tuple[str, list[str]]:
    source = get_source_by_key(mode, source_key)
    if not source:
        return f"The external source '{source_key}' is not registered for this project.", []

    source_description = source.get("description", source_key)
    link = source.get("link", "")

    if not link:
        return f"No URL is registered for external source '{source_key}'.", []

    console_log(mode, "EXTERNAL_QUERY", f"fetching url={link} for source_key={source_key}")
    try:
        fetched_content = await fetch_url_content(link)
    except Exception as exc:  # noqa: BLE001
        console_log(mode, "EXTERNAL_QUERY", f"fetch failed: {exc}")
        return f"Could not reach the external source '{source_key}' at {link}. Error: {exc}", []

    if not fetched_content.strip():
        return f"The external source '{source_key}' returned empty content.", []

    internal_data_json = "none"
    if internal_df is not None and not internal_df.empty:
        internal_rows = [{str(k): str(v) for k, v in row.items()} for _, row in internal_df.iterrows()]
        internal_data_json = json.dumps(internal_rows, ensure_ascii=False)

    system_prompt = prompt_loader.build_agent_system_prompt("final_answer_enrichment_agent")
    prompt = prompt_loader.render_task_prompt(
        "external_query_task",
        query=query,
        source_key=source_key,
        source_description=source_description,
        file_profiles="(content fetched live from URL)",
        data_slice=fetched_content,
        internal_data=internal_data_json,
    )

    try:
        result = await call_llm_guarded(
            call_llm,
            mode,
            system_prompt,
            prompt,
            call_timeout=get_call_timeout(mode),
            gemini_model=settings.gemini_model_enrichment,
        )
        await log_cost(
            user_id, session_id, answer_id,
            "external_query_agent",
            result.model, result.input_tokens, result.output_tokens, result.cached_tokens, mode,
        )
        console_log(mode, "EXTERNAL_QUERY", f"source_key={source_key} raw:\n{result.text}")
        decision = _extract_json(result.text)
        text = decision.get("text") or "I couldn't generate an answer from the external source."
        follow_ups = list(decision.get("followups") or [])[:2]
        return text, follow_ups
    except Exception as exc:  # noqa: BLE001
        console_log(mode, "EXTERNAL_QUERY", f"agent failed: {exc}")
        return "An error occurred while querying the external source. Please try again.", []


async def run_followups_only(
    *,
    mode: str,
    query: str,
    schema: dict,
    user_id: str,
    session_id: str,
    answer_id: str,
) -> tuple[Optional[str], list[str]]:
    system_prompt = prompt_loader.build_agent_system_prompt("final_answer_enrichment_agent")
    prompt = prompt_loader.render_task_prompt(
        "followups_task", query=query, schema=json.dumps(schema, ensure_ascii=False)
    )

    result = await call_llm_guarded(
        call_llm, mode, system_prompt, prompt, max_tokens=30000, call_timeout=get_call_timeout(mode), gemini_model=settings.gemini_model_enrichment
    )
    await log_cost(
        user_id,
        session_id,
        answer_id,
        "followups_only",
        result.model,
        result.input_tokens,
        result.output_tokens,
        result.cached_tokens,
        mode,
    )
    console_log(mode, "AGENT2_FOLLOWUPS", f"raw decision:\n{result.text}")

    try:
        payload = _extract_json(result.text)
        title = (payload.get("title") or "").strip() or None
        return title, list(payload.get("followups") or [])[:2]
    except (ValueError, json.JSONDecodeError):
        return None, []


async def run_rag_query_agent(
    *,
    mode: str,
    query: str,
    answer_id: str,
    user_id: str,
    session_id: str,
) -> tuple[str, list[str]]:
    from src.db import vector_store as _vs

    chunks = _vs.search(mode, query, top_k=6)
    if not chunks:
        return (
            "No relevant documents were found in the knowledge base for this project. "
            "Try uploading documents first via the Internal Knowledge section.",
            [],
        )

    context_block = "\n\n---\n\n".join(
        f"[{c['doc_name']}]\n{c['text']}" for c in chunks
    )

    system_prompt = (
        "You are a helpful knowledge assistant. Answer the user's question using "
        "ONLY the document excerpts provided. Be concise and accurate. "
        "If the answer is not in the excerpts, say so honestly. "
        "Always cite which document the information comes from."
    )
    user_prompt = (
        f"User question: {query!r}\n\n"
        f"Relevant document excerpts:\n\n{context_block}\n\n"
        "Reply with a JSON object: "
        '{"text": "<your answer>", "followups": ["<q1>", "<q2>"]}'
    )

    try:
        result = await call_llm_guarded(
            call_llm,
            mode,
            system_prompt,
            user_prompt,
            call_timeout=get_call_timeout(mode),
            gemini_model=settings.gemini_model_enrichment,
        )
        await log_cost(
            user_id, session_id, answer_id,
            "rag_query_agent",
            result.model, result.input_tokens, result.output_tokens, result.cached_tokens, mode,
        )
        console_log(mode, "RAG_QUERY", f"chunks_used={len(chunks)} raw:\n{result.text}")
        decision = _extract_json(result.text)
        text = decision.get("text") or "I couldn't generate an answer from the available documents."
        follow_ups = list(decision.get("followups") or [])[:2]
        return text, follow_ups
    except Exception as exc:  # noqa: BLE001
        console_log(mode, "RAG_QUERY", f"agent failed: {exc}")
        return "An error occurred while searching the knowledge base. Please try again.", []
