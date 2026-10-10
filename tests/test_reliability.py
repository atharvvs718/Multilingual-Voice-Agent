"""Retries, model fallback, the usage cap and the turn log's failure behaviour (unit 5)."""
import groq
import httpx
import psycopg
import pytest

import agent.llm as llm_module
import db.log as log_module
from agent.errors import ServiceUnavailable
from agent.limits import RateLimiter
from agent.retry import with_retries
from agent.turn import Turn
from config import LLM_FALLBACK_MODEL, LLM_MODEL
from tests.conftest import USAGE, fake_message

REQUEST = httpx.Request("POST", "https://api.groq.com/test")


def rate_limited() -> groq.RateLimitError:
    return groq.RateLimitError("rate limited", response=httpx.Response(429, request=REQUEST), body=None)


def test_retries_then_succeeds():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError
        return "ok"

    assert with_retries(flaky, (TimeoutError,)) == "ok"
    assert len(attempts) == 3


def test_gives_up_after_three_retries_and_reraises():
    attempts = []

    def always_fails():
        attempts.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        with_retries(always_fails, (TimeoutError,))
    assert len(attempts) == 4       # the first try plus three retries


def test_other_errors_are_not_retried():
    attempts = []

    def bug():
        attempts.append(1)
        raise KeyError("bug")

    with pytest.raises(KeyError):
        with_retries(bug, (TimeoutError,))
    assert len(attempts) == 1


def test_rate_limit_switches_to_fallback_model(monkeypatch):
    used = []

    def chat(model, messages, tools, max_tokens):
        used.append(model)
        if model == LLM_MODEL:
            raise rate_limited()
        return fake_message("ठीक आहे"), {**USAGE, "model": model}

    monkeypatch.setattr(llm_module, "_chat", chat)
    message, usage = llm_module.llm([{"role": "user", "content": "hi"}])
    assert used == [LLM_MODEL, LLM_FALLBACK_MODEL]      # no retries on the rate-limited model
    assert usage["model"] == LLM_FALLBACK_MODEL
    assert message.content == "ठीक आहे"


def test_all_models_down_raises_service_unavailable(monkeypatch):
    def chat(model, messages, tools, max_tokens):
        raise groq.APIConnectionError(request=REQUEST)

    monkeypatch.setattr(llm_module, "_chat", chat)
    with pytest.raises(ServiceUnavailable):
        llm_module.llm([{"role": "user", "content": "hi"}])


def test_explicit_model_has_no_fallback(monkeypatch):
    used = []

    def chat(model, messages, tools, max_tokens):
        used.append(model)
        raise rate_limited()

    monkeypatch.setattr(llm_module, "_chat", chat)
    with pytest.raises(ServiceUnavailable):
        llm_module.llm([{"role": "user", "content": "hi"}], model="some-model")
    assert used == ["some-model", "some-model"]     # one wait and one more try, but never another model


def test_rate_limiter_caps_and_recovers():
    now = [0.0]
    limiter = RateLimiter(max_events=2, window_s=60, clock=lambda: now[0])
    assert limiter.allow("a")
    assert limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")           # other visitors are unaffected
    now[0] = 61
    assert limiter.allow("a")           # the window has moved on


def test_rate_limiter_with_costs_is_a_budget():
    now = [0.0]
    budget = RateLimiter(max_events=500, window_s=3600, clock=lambda: now[0])
    assert budget.allow("everyone", cost=200)
    assert budget.allow("everyone", cost=300)       # exactly at the cap
    assert not budget.allow("everyone", cost=1)
    now[0] = 3601
    assert budget.allow("everyone", cost=400)       # the earlier spending has left the window
    assert not budget.allow("everyone", cost=101)
    assert not budget.allow("someone", cost=501)    # more than the whole budget is never allowed


def test_a_refused_event_is_not_counted():
    limiter = RateLimiter(max_events=300, window_s=60, clock=lambda: 0.0)
    assert not limiter.allow("a", cost=301)
    assert limiter.allow("a", cost=300)


def test_log_failure_does_not_raise(monkeypatch):
    def down(*_):
        raise psycopg.OperationalError("database is down")

    monkeypatch.setattr(log_module, "run", down)
    assert log_module.log_turn(Turn(session_id="s")) is None
    assert log_module.set_feedback(1, 1) is False


def test_feedback_value_is_validated():
    with pytest.raises(ValueError, match="1 or -1"):
        log_module.set_feedback(1, 5)


def test_waits_out_a_rate_limit_on_every_model(monkeypatch):
    calls = []

    def chat(model, messages, tools, max_tokens):
        calls.append(model)
        if len(calls) <= 2:                 # both models are over their per-minute limit at first
            raise rate_limited()
        return fake_message("ठीक आहे"), {**USAGE, "model": model}

    monkeypatch.setattr(llm_module, "_chat", chat)
    message, _ = llm_module.llm([{"role": "user", "content": "hi"}])
    assert calls == [LLM_MODEL, LLM_FALLBACK_MODEL, LLM_MODEL]
    assert message.content == "ठीक आहे"
