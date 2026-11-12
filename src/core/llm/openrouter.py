"""OpenRouter AI client — used exclusively in public mode.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint, so this
is a thin synchronous requests wrapper that returns the same LLMResult shape
as every other provider in this module.

OPEN_ROUTER_KEY lives in DS .env.  The backend never sends it.
"""
import requests

from src.config import settings
from src.core.llm.client import LLMProviderError, LLMResult, TokenLimitExceededError


def _is_token_limit(text: str) -> bool:
    low = text.lower()
    return "token" in low and ("exceed" in low or "too long" in low or "too many" in low)


def _map_status(status: int, body: str) -> str:
    low = body.lower()
    if status == 400:
        return "invalid_request"
    if status == 401:
        return "authentication"
    if status == 403:
        return "permission_denied"
    if status == 404:
        return "model_not_found"
    if status == 429:
        return "quota_exceeded" if "quota" in low else "rate_limit_exceeded"
    if status == 500:
        return "api_error"
    if status in (502, 503):
        return "service_unavailable"
    if status == 504:
        return "deadline_exceeded"
    return "api_error"


def call_openrouter(system: str, user: str, model: str, max_tokens: int) -> LLMResult:
    """Submit a chat-completion request to OpenRouter and return an LLMResult."""
    api_key = settings.open_router_key
    if not api_key:
        raise LLMProviderError("authentication", "OPEN_ROUTER_KEY is not configured in the DS environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise LLMProviderError("deadline_exceeded", "OpenRouter request timed out") from exc
    except requests.exceptions.ConnectionError as exc:
        raise LLMProviderError("service_unavailable", f"Could not reach OpenRouter: {exc}") from exc
    except requests.exceptions.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        if _is_token_limit(body):
            raise TokenLimitExceededError(body) from exc
        code = _map_status(exc.response.status_code, body)
        raise LLMProviderError(code, body) from exc

    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    content = choice["message"].get("content") or ""
    if not content:
        # Some free models return null content (e.g. refusals, tool-call-only
        # responses). Treat it as an empty response so callers get a string.
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "OpenRouter returned null/empty content for model=%s finish_reason=%s",
            model,
            choice.get("finish_reason"),
        )
    return LLMResult(
        text=content,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=model,
    )
