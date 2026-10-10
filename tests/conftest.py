"""Shared fakes. No test touches Groq, Sarvam, the GPU or the database."""
from types import SimpleNamespace

import pytest

USAGE = {"model": "fake-model", "tokens_in": 100, "tokens_out": 10}


def fake_message(content: str = "", tool: str | None = None, arguments: str = "{}"):
    """Shaped like the message object the Groq SDK returns."""
    calls = None
    if tool:
        calls = [SimpleNamespace(id="call_1", function=SimpleNamespace(name=tool, arguments=arguments))]
    return SimpleNamespace(content=content, tool_calls=calls)


class ScriptedLLM:
    """Stands in for agent.llm.llm: returns the given messages in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def __call__(self, messages, tools=None, **_):
        self.calls.append(([dict(m) for m in messages], tools))
        return self.replies.pop(0), dict(USAGE)


def memory_budget(rupees_per_day: float, already: float = 0.0):
    """A day's budget counted in a list instead of the database, behaving as the real statement does."""
    from agent.budget import Budget

    spent = [already]

    def charge(day, service, rupees, limit) -> bool:
        if spent[0] + rupees > limit:
            return False
        spent[0] += rupees
        return True

    return Budget(rupees_per_day, charge=charge, spent=lambda day: spent[0])


@pytest.fixture(autouse=True)
def budget_in_memory(monkeypatch):
    """A fresh day's budget for every test, counted in memory instead of in the database."""
    from agent import budget

    monkeypatch.setattr(budget, "daily", memory_budget(30))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retries back off with time.sleep; tests should not wait for it."""
    monkeypatch.setattr("agent.retry.time.sleep", lambda _seconds: None)
