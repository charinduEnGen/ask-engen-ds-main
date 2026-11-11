"""Analytics query pipeline — entry point for mode="analytics" queries.

Moved from src/orchestrator/query_orchestrator.py; that file is now a
backward-compat re-export.
"""

import asyncio
import time
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks

_COMPARISON_KEYWORDS = frozenset({
    "compare", "comparison", "versus", "vs", "contrast",
    "difference", "diff", "gap", "deviation", "against",
})


def _is_comparison_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _COMPARISON_KEYWORDS)


from src.config import settings
from src.db.engine import get_engine
from src.services.analytics import steps
from src.core.llm.gate import _llm_semaphore  # noqa: F401 - re-exported; see gate module docstring
from src.services.analytics.steps import Agent1Outcome, IntentionOutcome, run_rag_query_agent
from src.schemas.user_request import QueryResponse, UserRequest
from src.services.analytics.tools import response_builder
from src.utils import kb_loader, masking
from src.core.session import memory as session_memory
from src.utils.id_utils import generate_answer_id, generate_question_id
from src.core.llm.client import ERROR_FRIENDLY_MESSAGES, LLMProviderError, TokenLimitExceededError
from src.utils.logging_utils import StageTimer, console_log, log_agent1_failure, log_timing

_TOKEN_LIMIT_MESSAGE = (
    "This session's conversation history has grown too large for me to "
    "process alongside your question. Could you start a new session or "
    "ask a more focused question?"
)


def _provider_error_message(exc: LLMProviderError) -> str:
    return ERROR_FRIENDLY_MESSAGES.get(exc.code, ERROR_FRIENDLY_MESSAGES["api_error"])


def _log_masking(mode: str, stage: str, mask_key: masking.MaskKey) -> None:
    if mask_key.is_empty():
        console_log(mode, "MASKING", f"{stage}: no sensitive columns present, nothing to mask")
    else:
        console_log(mode, "MASKING", f"{stage}: key={mask_key.key_id} column(s)={mask_key.columns}")


async def run_query_pipeline(
    user_request: UserRequest, background_tasks: BackgroundTasks
) -> QueryResponse:
    start = time.perf_counter()
    timer = StageTimer()
    mode = user_request.mode
    answer_id = generate_answer_id()
    question_id = user_request.question_id or generate_question_id()

    console_log(
        mode,
        "REQUEST",
        f"answer_id={answer_id} question_id={question_id} user_id={user_request.user_id} "
        f"session_id={user_request.session_id} query={user_request.query!r}",
    )

    effective_query = user_request.query

    chat_expire = session_memory.is_chat_expired(user_request.user_id, user_request.session_id, mode)
    max_context_window = session_memory.is_context_window_full(user_request.user_id, user_request.session_id, mode)

    if chat_expire or max_context_window:
        console_log(
            mode,
            "ROUTE",
            f"chat_expire={chat_expire} max_context_window={max_context_window} — "
            "skipping pipeline, returning empty answer envelope",
        )
        response = response_builder.build_response(
            answer_id=answer_id,
            question_id=question_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            max_context_window=max_context_window,
            chat_expire=chat_expire,
            text_content=None,
            table=None,
            chart=None,
            follow_ups=[],
            merchant_disambiguation=None,
        )
        total_seconds = round(time.perf_counter() - start, 4)
        console_log(
            mode,
            "RESPONSE",
            f"shape={_response_shape(response)} maxContextWindow={max_context_window} "
            f"chatExpire={chat_expire} total_seconds={total_seconds} answer_id={answer_id}",
        )
        background_tasks.add_task(
            log_timing,
            user_request.user_id,
            user_request.session_id,
            answer_id,
            effective_query,
            timer.stages,
            total_seconds,
            mode,
        )
        return response

    token_limit_hit = False
    provider_error_message: Optional[str] = None

    engine = get_engine(mode)

    with timer.track("build_context"):
        schema = kb_loader.load_schema(mode)
        analytics_patterns = kb_loader.load_analytics_patterns(mode)
        data_context = kb_loader.render_prompt_summary(mode)
        history = session_memory.get_history(user_request.user_id, user_request.session_id, mode)
        memory_context = session_memory.format_history_for_prompt(history)

    console_log(mode, "MEMORY", f"{len(history)} stored turn(s) for this session:\n{memory_context}")

    with timer.track("intention_classification"):
        intention = await steps.run_intention_classifier(
            mode=mode,
            query=effective_query,
            memory_context=memory_context,
            answer_id=answer_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
        )

    if intention.intent == "rag":
        console_log(mode, "ROUTE", "rag intent detected — searching knowledge base")
        with timer.track("rag_query"):
            try:
                text_content, follow_ups = await run_rag_query_agent(
                    mode=mode,
                    query=effective_query,
                    answer_id=answer_id,
                    user_id=user_request.user_id,
                    session_id=user_request.session_id,
                )
            except Exception as exc:  # noqa: BLE001
                console_log(mode, "RAG_QUERY", f"agent raised: {exc}")
                text_content = "An error occurred searching the knowledge base."
                follow_ups = []

        response = response_builder.build_response(
            answer_id=answer_id,
            question_id=question_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            max_context_window=False,
            chat_expire=False,
            text_content=text_content,
            table=None,
            chart=None,
            follow_ups=follow_ups or None,
            merchant_disambiguation=None,
        )
        total_seconds = round(time.perf_counter() - start, 4)
        console_log(
            mode, "RESPONSE",
            f"shape={_response_shape(response)} rag_query "
            f"stages={timer.stages} total_seconds={total_seconds} answer_id={answer_id}",
        )
        background_tasks.add_task(
            session_memory.append_turn,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            answer_id=answer_id,
            query=effective_query,
            answer=response.answer.model_dump(exclude_none=True),
            response_type=_response_shape(response),
            mask_map=None,
            pending_disambiguation=None,
        )
        background_tasks.add_task(
            log_timing,
            user_request.user_id, user_request.session_id,
            answer_id, effective_query, timer.stages, total_seconds, mode,
        )
        return response

    if intention.intent == "external" and intention.source_key:
        console_log(mode, "ROUTE", f"external intent detected: source_key={intention.source_key}")

        internal_df: pd.DataFrame | None = None
        if _is_comparison_query(effective_query):
            console_log(mode, "ROUTE", "comparison query detected — fetching internal data for cross-source diff")
            with timer.track("internal_for_comparison"):
                try:
                    internal_outcome = await steps.run_data_processing_agent(
                        mode=mode,
                        query=effective_query,
                        schema=schema,
                        data_context=data_context,
                        analytics_patterns=analytics_patterns,
                        memory_context=memory_context,
                        engine=engine,
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                        answer_id=answer_id,
                    )
                    if internal_outcome.kind == "result" and not internal_outcome.result_df.empty:
                        internal_df = internal_outcome.result_df
                        console_log(mode, "ROUTE", f"internal data fetched: {len(internal_df)} row(s)")
                except Exception as exc:  # noqa: BLE001
                    console_log(mode, "ROUTE", f"internal fetch for comparison failed (non-fatal): {exc}")

        with timer.track("external_query"):
            try:
                text_content, follow_ups = await steps.run_external_query_agent(
                    mode=mode,
                    query=effective_query,
                    source_key=intention.source_key,
                    answer_id=answer_id,
                    user_id=user_request.user_id,
                    session_id=user_request.session_id,
                    internal_df=internal_df,
                )
            except Exception as exc:  # noqa: BLE001
                console_log(mode, "EXTERNAL_QUERY", f"agent raised: {exc}")
                text_content = "An error occurred reading the external data source."
                follow_ups = []

        mask_key = masking.MaskKey(
            key_id=f"{user_request.user_id}:{user_request.session_id}:{mode}:{question_id}:{answer_id}"
        )
        response = response_builder.build_response(
            answer_id=answer_id,
            question_id=question_id,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            max_context_window=False,
            chat_expire=False,
            text_content=text_content,
            table=None,
            chart=None,
            follow_ups=follow_ups or None,
            merchant_disambiguation=None,
        )
        total_seconds = round(time.perf_counter() - start, 4)
        console_log(
            mode, "RESPONSE",
            f"shape={_response_shape(response)} external_source={intention.source_key} "
            f"stages={timer.stages} total_seconds={total_seconds} answer_id={answer_id}",
        )
        memory_answer = response.answer.model_dump(exclude_none=True)
        background_tasks.add_task(
            session_memory.append_turn,
            user_id=user_request.user_id,
            session_id=user_request.session_id,
            mode=mode,
            answer_id=answer_id,
            query=effective_query,
            answer=memory_answer,
            response_type=_response_shape(response),
            mask_map=None,
            pending_disambiguation=None,
        )
        background_tasks.add_task(
            log_timing,
            user_request.user_id, user_request.session_id,
            answer_id, effective_query, timer.stages, total_seconds, mode,
        )
        return response

    reply_to_mask_key = None
    if user_request.answerId:
        prior_mask_map = session_memory.get_mask_map_for_answer(
            user_request.user_id, user_request.session_id, mode, user_request.answerId
        )
        if prior_mask_map:
            reply_to_mask_key = masking.from_storage(prior_mask_map, key_id=f"reply_to:{user_request.answerId}")
            console_log(
                mode,
                "MASKING",
                f"loaded mask key from answerId={user_request.answerId} column(s)={reply_to_mask_key.columns}",
            )

    try:
        with timer.track("agent1_query_build_and_execute"):
            outcome = await steps.run_data_processing_agent(
                mode=mode,
                query=effective_query,
                schema=schema,
                data_context=data_context,
                analytics_patterns=analytics_patterns,
                memory_context=memory_context,
                engine=engine,
                user_id=user_request.user_id,
                session_id=user_request.session_id,
                reply_to_mask_key=reply_to_mask_key,
                answer_id=answer_id,
            )
    except TokenLimitExceededError:
        token_limit_hit = True
        outcome = Agent1Outcome(
            kind="out_of_scope", out_of_scope_reason="token limit exceeded on agent 1", scope_category="system_error"
        )
        console_log(mode, "TOKEN_LIMIT", "agent 1 aborted: token limit exceeded for this turn")
    except LLMProviderError as exc:
        provider_error_message = _provider_error_message(exc)
        outcome = Agent1Outcome(
            kind="out_of_scope", out_of_scope_reason=f"provider error ({exc.code}) on agent 1", scope_category="system_error"
        )
        console_log(mode, "PROVIDER_ERROR", f"agent 1 aborted: {exc.code} — {exc}")
    except asyncio.TimeoutError:
        provider_error_message = ERROR_FRIENDLY_MESSAGES["deadline_exceeded"]
        outcome = Agent1Outcome(
            kind="out_of_scope", out_of_scope_reason="LLM call timed out on agent 1", scope_category="system_error"
        )
        console_log(mode, "TIMEOUT", "agent 1 aborted: LLM call timed out")

    if outcome.kind == "result":
        console_log(
            mode,
            "AGENT1_RESULT",
            f"query executed for project '{mode}':\n{outcome.query_used}\n"
            f"-> {len(outcome.result_df)} row(s), {len(outcome.result_df.columns)} column(s)",
        )
    else:
        console_log(mode, "AGENT1_RESULT", f"kind={outcome.kind}")

    mask_key = masking.MaskKey(key_id=f"{user_request.user_id}:{user_request.session_id}:{mode}:{question_id}:{answer_id}")

    if outcome.kind != "result":
        if token_limit_hit:
            console_log(mode, "ROUTE", "token limit hit on agent 1 — skipping response-building agent(s)")
            text_content = _TOKEN_LIMIT_MESSAGE
        elif provider_error_message:
            console_log(mode, "ROUTE", "provider error hit on agent 1 — skipping response-building agent(s)")
            text_content = provider_error_message
        else:
            console_log(
                mode,
                "ROUTE",
                f"out of scope (category={outcome.scope_category}) / no usable code — "
                "running out-of-scope response agent",
            )
            with timer.track("out_of_scope_response"):
                try:
                    text_content = await steps.run_out_of_scope_response_agent(
                        mode=mode,
                        query=effective_query,
                        scope_category=outcome.scope_category or "unsupported_data",
                        out_of_scope_reason=outcome.out_of_scope_reason or "no reason given",
                        answer_id=answer_id,
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                    )
                except TokenLimitExceededError:
                    token_limit_hit = True
                    console_log(mode, "TOKEN_LIMIT", "out-of-scope response agent aborted: token limit exceeded for this turn")
                    text_content = _TOKEN_LIMIT_MESSAGE
                except LLMProviderError as exc:
                    console_log(mode, "PROVIDER_ERROR", f"out-of-scope response agent aborted: {exc.code} — {exc}")
                    text_content = _provider_error_message(exc)
                except asyncio.TimeoutError:
                    console_log(mode, "TIMEOUT", "out-of-scope response agent aborted: LLM call timed out")
                    text_content = ERROR_FRIENDLY_MESSAGES["deadline_exceeded"]
        table = None
        chart = None
        follow_ups = None
    else:
        result_df = outcome.result_df
        row_count = len(result_df)
        with timer.track("build_response"):
            try:
                if row_count <= settings.small_result_threshold:
                    console_log(mode, "ROUTE", f"{row_count} row(s) <= small_result_threshold: running enrichment agent")
                    mask_key = masking.build_mask_key(
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                        mode=mode,
                        question_id=question_id,
                        answer_id=answer_id,
                        df=result_df,
                        secure_cols=masking.secure_columns_for(schema),
                    )
                    _log_masking(mode, "mask before enrichment prompt", mask_key)
                    masked_result_df = masking.mask_dataframe(result_df, mask_key)
                    text_content, table, chart, follow_ups = await steps.run_enrichment_agent(
                        mode=mode,
                        query=effective_query,
                        schema=schema,
                        result_df=masked_result_df,
                        answer_id=answer_id,
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                    )
                    text_content = masking.demask_json(text_content, mask_key)
                    table = masking.demask_json(table, mask_key)
                    chart = masking.demask_json(chart, mask_key)
                    follow_ups = masking.demask_json(follow_ups, mask_key)
                    _log_masking(mode, "demask before returning to caller", mask_key)
                else:
                    console_log(mode, "ROUTE", f"{row_count} row(s) > small_result_threshold: running follow-ups-only agent + full table")
                    text_content = None
                    chart = None
                    table_title, follow_ups = await steps.run_followups_only(
                        mode=mode,
                        query=effective_query,
                        schema=schema,
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                        answer_id=answer_id,
                    )
                    table = response_builder.build_table_payload(
                        result_df,
                        page=1,
                        page_size=settings.default_page_size,
                        paginated=False,
                        title=table_title or effective_query,
                    )
                    mask_key = masking.build_mask_key(
                        user_id=user_request.user_id,
                        session_id=user_request.session_id,
                        mode=mode,
                        question_id=question_id,
                        answer_id=answer_id,
                        df=pd.DataFrame(table["rows"]),
                        secure_cols=masking.secure_columns_for(schema),
                    )
                    _log_masking(mode, "mask key scoped to visible page (table only, no LLM saw real data)", mask_key)
            except TokenLimitExceededError:
                token_limit_hit = True
                console_log(mode, "TOKEN_LIMIT", "response-building agent aborted: token limit exceeded for this turn")
                text_content = _TOKEN_LIMIT_MESSAGE
                table = None
                chart = None
                follow_ups = None
            except LLMProviderError as exc:
                console_log(mode, "PROVIDER_ERROR", f"response-building agent aborted: {exc.code} — {exc}")
                text_content = _provider_error_message(exc)
                table = None
                chart = None
                follow_ups = None
            except asyncio.TimeoutError:
                console_log(mode, "TIMEOUT", "response-building agent aborted: LLM call timed out")
                text_content = ERROR_FRIENDLY_MESSAGES["deadline_exceeded"]
                table = None
                chart = None
                follow_ups = None

    max_context_window = max_context_window or token_limit_hit

    response = response_builder.build_response(
        answer_id=answer_id,
        question_id=question_id,
        user_id=user_request.user_id,
        session_id=user_request.session_id,
        mode=mode,
        max_context_window=max_context_window,
        chat_expire=chat_expire,
        text_content=text_content,
        table=table,
        chart=chart,
        follow_ups=follow_ups,
        merchant_disambiguation=None,
    )

    total_seconds = round(time.perf_counter() - start, 4)

    console_log(
        mode,
        "RESPONSE",
        f"shape={_response_shape(response)} maxContextWindow={max_context_window} "
        f"chatExpire={chat_expire} stages={timer.stages} total_seconds={total_seconds} "
        f"answer_id={answer_id}",
    )

    memory_answer = response.answer.model_dump(exclude_none=True)
    memory_answer = masking.mask_json(memory_answer, mask_key)
    _log_masking(mode, "mask before session-memory write", mask_key)

    background_tasks.add_task(
        session_memory.append_turn,
        user_id=user_request.user_id,
        session_id=user_request.session_id,
        mode=mode,
        answer_id=answer_id,
        query=effective_query,
        answer=memory_answer,
        response_type=_response_shape(response),
        mask_map=masking.to_storage(mask_key) if not mask_key.is_empty() else None,
        pending_disambiguation=None,
    )
    background_tasks.add_task(
        log_timing,
        user_request.user_id,
        user_request.session_id,
        answer_id,
        effective_query,
        timer.stages,
        total_seconds,
        mode,
    )
    if outcome.failed_attempts:
        background_tasks.add_task(
            log_agent1_failure,
            user_request.user_id,
            user_request.session_id,
            answer_id,
            effective_query,
            mode,
            outcome.scope_category or "system_error",
            outcome.out_of_scope_reason or "",
            outcome.failed_attempts,
        )

    return response


def _response_shape(response: QueryResponse) -> str:
    if response.answer.chart:
        return "text_table_chart"
    if response.answer.table:
        return "text_table" if response.answer.text.get("content") else "table"
    return "text"
