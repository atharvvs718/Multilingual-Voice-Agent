"""Every LLM call in the project goes through llm() or see(). Model names appear only in config."""
import base64
import logging
import time
from functools import lru_cache

import groq

from agent.errors import ServiceUnavailable
from agent.retry import with_retries
from config import API_TIMEOUT_S, GROQ_API_KEY, LLM_FALLBACK_MODEL, LLM_MODEL, VISION_MODEL

log = logging.getLogger(__name__)

RATE_LIMIT_WAIT_S = 8.0     # the longest a turn will wait for the free tier's per-minute limit to clear
# How long to wait for the main model before handing the call to the smaller one. The free tier gives
# 8,000 tokens a minute and one turn can use most of it, so the limit is often a few seconds from clearing;
# the smaller model answers at once but noticeably worse.
PATIENCE_S = 6.0
VISION_TIMEOUT_S = 40

# Failures worth retrying: the network, a timeout, or a 5xx from Groq.
TRANSIENT = (groq.APIConnectionError, groq.APITimeoutError, groq.InternalServerError)


class ToolCallGlitch(Exception):
    """The model produced a malformed tool call; asking again usually works."""


@lru_cache(maxsize=1)
def groq_client() -> groq.Groq:
    if not GROQ_API_KEY:
        raise ServiceUnavailable("GROQ_API_KEY is missing. Put it in .env")
    # max_retries=0: retries are ours (with_retries), so they are logged and counted in one place.
    return groq.Groq(api_key=GROQ_API_KEY, max_retries=0, timeout=API_TIMEOUT_S)


def _chat(model: str, messages: list[dict], tools: list[dict] | None, max_tokens: int):
    kwargs = {"model": model, "messages": messages, "temperature": 0.2, "max_completion_tokens": max_tokens}
    if "gpt-oss" in model:
        # gpt-oss thinks before answering and that thinking is billed as output tokens; keep it short.
        kwargs["reasoning_effort"] = "low"
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    try:
        response = groq_client().chat.completions.create(**kwargs)
    except groq.BadRequestError as error:
        if "tool_use_failed" in str(error):
            raise ToolCallGlitch(str(error)) from error
        raise
    usage = {
        "model": model,
        "tokens_in": response.usage.prompt_tokens,
        "tokens_out": response.usage.completion_tokens,
    }
    return response.choices[0].message, usage


def llm(messages: list[dict], tools: list[dict] | None = None, model: str | None = None, max_tokens: int = 700):
    """One chat call. Returns (message, usage) where usage has model, tokens_in and tokens_out.

    The model remembers nothing between calls: messages must carry everything it needs.
    With no model given, the default is tried first and the fallback (which has its own
    free-tier quota) takes over on a rate limit or repeated failure (FR-11).
    """
    models = [model] if model else [LLM_MODEL, LLM_FALLBACK_MODEL]
    last_error: Exception | None = None
    for attempt in (1, 2):
        for name in models:
            try:
                return with_retries(
                    lambda name=name: _chat(name, messages, tools, max_tokens), (*TRANSIENT, ToolCallGlitch)
                )
            except groq.RateLimitError as error:
                last_error = error
                wait = _retry_after(error)
                if attempt == 1 and name == LLM_MODEL and len(models) > 1 and wait <= PATIENCE_S:
                    log.warning("%s is rate limited for %.1f s; waiting for it", name, wait)
                    time.sleep(wait)
                    try:
                        return _chat(name, messages, tools, max_tokens)
                    except (groq.RateLimitError, *TRANSIENT, ToolCallGlitch) as again:
                        last_error = again
                log.warning("%s is rate limited; trying the next model", name)
            except (*TRANSIENT, ToolCallGlitch) as error:
                log.warning("%s kept failing (%s); trying the next model", name, type(error).__name__)
                last_error = error
        if attempt == 1 and isinstance(last_error, groq.RateLimitError):
            # Every model is over its per-minute limit. That clears in seconds, so wait once and go round again.
            wait = min(_retry_after(last_error), RATE_LIMIT_WAIT_S)
            log.warning("all models are rate limited; waiting %.1f s", wait)
            time.sleep(wait)
        else:
            break
    raise ServiceUnavailable(f"LLM unavailable: {type(last_error).__name__}") from last_error


def see(request: str, pictures: list[bytes], max_tokens: int = 1200) -> str:
    """Ask the model that can read pictures about one or more JPEG images; returns its answer.

    Used to read a page of the student's book. It has its own free-tier quota, separate from the main model's.
    """
    content = [{"type": "text", "text": request}]
    for picture in pictures:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(picture).decode()}})

    def request_once():
        return groq_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_completion_tokens=max_tokens,
            timeout=VISION_TIMEOUT_S,
        )

    try:
        response = with_retries(request_once, TRANSIENT)
    except (*TRANSIENT, groq.RateLimitError, groq.BadRequestError) as error:
        raise ServiceUnavailable(f"the page could not be read: {type(error).__name__}") from error
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise ServiceUnavailable("the page could not be read: empty answer")
    return answer


def _retry_after(error: groq.RateLimitError) -> float:
    """Seconds the API asks us to wait, from its retry-after header; the maximum wait if it does not say."""
    try:
        return float(error.response.headers.get("retry-after", RATE_LIMIT_WAIT_S))
    except (AttributeError, TypeError, ValueError):
        return RATE_LIMIT_WAIT_S
