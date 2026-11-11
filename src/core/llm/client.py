"""Thin, provider-agnostic LLM wrapper shared by every agent in the pipeline.

Switch providers with the LLM_PROVIDER env var (anthropic | gemini) — no
call-site changes needed. This is the definitive location; src/utils/llm_client.py
is a backward-compat re-export.
"""

from dataclasses import dataclass
from typing import Optional

from src.config import settings
from src.utils.logging_utils import console_log


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached_tokens: int = 0


class TokenLimitExceededError(RuntimeError):
    """Raised when the prompt exceeds the model's token limit."""


class LLMProviderError(RuntimeError):
    """Raised when the provider API rejects a call. `code` is always one of
    ERROR_FRIENDLY_MESSAGES' keys."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


ERROR_FRIENDLY_MESSAGES: dict[str, str] = {
    "invalid_request": "I couldn't process that request — could you rephrase your question?",
    "failed_precondition": "This assistant isn't available to answer right now due to a service configuration issue. Please try again shortly.",
    "out_of_range": "That request is outside what I'm able to handle — could you narrow it down?",
    "parameter_unknown": "I couldn't process that request — could you rephrase your question?",
    "authentication": "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "permission_denied": "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "not_found": "I couldn't find what I needed to answer that — could you rephrase your question?",
    "model_not_found": "I'm temporarily unable to reach the underlying service. Please try again shortly.",
    "already_exists": "Something conflicted while handling your request — please try again.",
    "aborted": "Something conflicted while handling your request — please try again.",
    "rate_limit_exceeded": "I'm getting a lot of requests right now — please wait a moment and try again.",
    "quota_exceeded": "I've hit today's usage limit and can't answer right now. Please try again later.",
    "cancelled": "The request was cancelled before I could finish. Please try again.",
    "api_error": "Something went wrong on my end. Please try again in a moment.",
    "unimplemented": "That's not something I'm able to do.",
    "service_unavailable": "The service is temporarily overloaded right now. Please try again in a moment.",
    "deadline_exceeded": "I wasn't able to finish processing that in time. Please try again",
}
_DEFAULT_ERROR_CODE = "api_error"


def _map_gemini_exception(exc: "GoogleAPICallError") -> LLMProviderError:
    message = str(exc)
    lowered = message.lower()
    status = getattr(exc, "code", None)

    if status == 400:
        if "unknown" in lowered and "parameter" in lowered:
            code = "parameter_unknown"
        elif type(exc).__name__ == "FailedPrecondition":
            code = "failed_precondition"
        else:
            code = "invalid_request"
    elif status == 401:
        code = "authentication"
    elif status == 403:
        code = "permission_denied"
    elif status == 404:
        code = "model_not_found" if "model" in lowered else "not_found"
    elif status == 409:
        code = "already_exists" if type(exc).__name__ == "AlreadyExists" else "aborted"
    elif status == 412:
        code = "failed_precondition"
    elif status == 416:
        code = "out_of_range"
    elif status == 429:
        code = "quota_exceeded" if "quota" in lowered else "rate_limit_exceeded"
    elif status == 499:
        code = "cancelled"
    elif status == 501:
        code = "unimplemented"
    elif status in (502, 503):
        code = "service_unavailable"
    elif status == 504:
        code = "deadline_exceeded"
    else:
        code = _DEFAULT_ERROR_CODE
    return LLMProviderError(code, message)


def _map_anthropic_exception(exc: Exception) -> LLMProviderError:
    import anthropic

    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, anthropic.APITimeoutError):
        return LLMProviderError("deadline_exceeded", message)
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMProviderError("service_unavailable", message)

    status = getattr(exc, "status_code", None)
    if status == 400:
        code = "invalid_request"
    elif status == 401:
        code = "authentication"
    elif status == 403:
        code = "permission_denied"
    elif status == 404:
        code = "model_not_found" if "model" in lowered else "not_found"
    elif status == 409:
        code = "aborted"
    elif status == 429:
        code = "quota_exceeded" if "quota" in lowered else "rate_limit_exceeded"
    elif status == 500:
        code = "api_error"
    elif status == 529:
        code = "service_unavailable"
    else:
        code = _DEFAULT_ERROR_CODE
    return LLMProviderError(code, message)


def _call_anthropic(system: str, user: str, max_tokens: int) -> LLMResult:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        response = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=max_tokens,
            thinking={"type": "disabled"},
            output_config={"effort": "medium"},
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
        if _is_token_limit_error(str(exc)):
            raise TokenLimitExceededError(str(exc)) from exc
        raise _map_anthropic_exception(exc) from exc
    text = "".join(block.text for block in response.content if block.type == "text")
    return LLMResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=settings.anthropic_model,
    )


def _is_token_limit_error(message: str) -> bool:
    lowered = message.lower()
    return "token" in lowered and ("exceed" in lowered or "too long" in lowered or "too many" in lowered)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _call_gemini(
    mode: str,
    system: str,
    user: str,
    max_tokens: int,
    model_name: str,
    static_context: Optional[str],
) -> LLMResult:
    import google.generativeai as genai
    from google.api_core.exceptions import GoogleAPICallError

    genai.configure(api_key=settings.gemini_api_key)

    cached_content = None
    if static_context and settings.gemini_context_cache_enabled:
        from src.core.llm import gemini_cache

        cached_content = gemini_cache.get_or_create_cache(
            mode=mode,
            model_name=model_name,
            system_instruction=system,
            static_context=static_context,
            ttl_seconds=settings.gemini_cache_ttl_seconds,
        )

    if cached_content is not None:
        model = genai.GenerativeModel.from_cached_content(cached_content=cached_content)
    else:
        model = genai.GenerativeModel(model_name, system_instruction=system)
        if static_context:
            user = f"{static_context}\n\n{user}"

    try:
        response = model.generate_content(
            user, generation_config={"max_output_tokens": max_tokens}
        )
    except GoogleAPICallError as exc:
        if _is_token_limit_error(str(exc)):
            raise TokenLimitExceededError(str(exc)) from exc
        raise _map_gemini_exception(exc) from exc
    usage = response.usage_metadata
    return LLMResult(
        text=response.text,
        input_tokens=usage.prompt_token_count,
        output_tokens=usage.candidates_token_count,
        model=model_name,
        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
    )


def call_llm(
    mode: str,
    system: str,
    user: str,
    max_tokens: Optional[int] = None,
    gemini_model: Optional[str] = None,
    gemini_static_context: Optional[str] = None,
) -> LLMResult:
    max_tokens = max_tokens or settings.llm_max_output_tokens

    estimated_input_tokens = (
        _estimate_tokens(system) + _estimate_tokens(user) + _estimate_tokens(gemini_static_context or "")
    )
    console_log(
        mode,
        "LLM_CALL",
        f"estimated input ~{estimated_input_tokens} tokens (cap {settings.llm_max_input_tokens}), "
        f"output cap {max_tokens}, provider={settings.llm_provider}",
    )
    if estimated_input_tokens > settings.llm_max_input_tokens:
        raise TokenLimitExceededError(
            f"Estimated input size (~{estimated_input_tokens} tokens) exceeds "
            f"llm_max_input_tokens ({settings.llm_max_input_tokens})"
        )

    if settings.llm_provider == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    if settings.llm_provider == "gemini":
        return _call_gemini(
            mode, system, user, max_tokens, gemini_model or settings.gemini_model_data_processing, gemini_static_context
        )
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
