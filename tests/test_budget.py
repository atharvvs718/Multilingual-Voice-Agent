"""The daily money budget, and what the app does once it is used up."""
from datetime import date

import numpy as np
import psycopg
import pytest
from fastapi.testclient import TestClient

import agent.sarvam as sarvam_api
import agent.stt as stt
import server
from agent import budget
from agent.budget import Budget
from rag.ingest import IngestResult
from tests.conftest import memory_budget
from tests.test_server import api, ask  # noqa: F401  (api is a fixture)


def test_spending_stops_at_the_days_budget():
    wallet = memory_budget(30)
    assert wallet.spend(20, "voice")
    assert wallet.spend(10, "listening")            # exactly at the budget
    assert not wallet.spend(0.01, "voice")
    assert wallet.spent() == 30


def test_what_was_already_spent_today_counts():
    wallet = memory_budget(30, already=29.5)        # as after a restart, or with another copy of the app running
    assert not wallet.spend(1, "voice")
    assert wallet.spend(0.5, "voice")


def test_nothing_paid_is_used_when_the_days_spending_cannot_be_checked():
    def down(*_):
        raise psycopg.OperationalError("database is down")

    blind = Budget(30, charge=down, spent=down, today=lambda: date(2026, 10, 4))
    assert not blind.spend(0.1, "voice")
    assert blind.spent() == 30


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0])
def test_a_cost_that_is_not_a_real_amount_is_refused(cost):
    wallet = memory_budget(30)
    assert not wallet.spend(cost, "voice")
    assert wallet.spent() == 0


def test_the_charge_is_one_statement_that_checks_and_adds_together():
    statement = " ".join(budget._CHARGE.split())
    # the limit is tested on the total's own row while the database holds it, not on a figure read beforehand
    assert "DO UPDATE SET rupees = spend.rupees + excluded.rupees WHERE spend.rupees + excluded.rupees <= %(limit)s" in statement
    assert "SELECT %(day)s, 'total', %(rupees)s WHERE %(rupees)s <= %(limit)s" in statement     # also for the day's first charge
    assert "SELECT %(day)s, %(service)s, %(rupees)s FROM counted" in statement      # the service's row follows only a granted charge


# --- listening --------------------------------------------------------------------------

def test_over_the_budget_listening_moves_to_the_free_model(monkeypatch):
    used = []
    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    monkeypatch.setattr(stt, "_sarvam", lambda audio, hint: used.append("sarvam") or ("ok", "mr"))
    monkeypatch.setattr(stt, "_whisper", lambda audio, hint: used.append("whisper") or ("ok", "marathi"))
    five_seconds = np.zeros(16000 * 5, dtype=np.float32)
    cost = 5 * budget.LISTENING_RUPEES_PER_SECOND
    monkeypatch.setattr(budget, "daily", memory_budget(cost * 2.5))
    for _ in range(3):
        stt.transcribe(five_seconds, "mr")
    assert used == ["sarvam", "sarvam", "whisper"]


# --- the voice --------------------------------------------------------------------------

def test_over_the_budget_the_page_is_told_and_given_the_words_to_say_itself(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(budget, "daily", memory_budget(0.2))       # enough for one short reply
    first = ask(api, text="प्रश्न")[0]
    assert api.post("/api/speak", json={"chat": first["chat"]["id"], "key": first["message"]["key"]}).status_code == 200
    second = ask(api, text="प्रश्न")[0]
    where = {"chat": second["chat"]["id"], "key": second["message"]["key"]}
    refused = api.post("/api/speak", json=where)
    assert refused.status_code == 429
    body = refused.json()
    assert body["message"].startswith("Today's voice budget is used up")
    assert body["speech"] == {"text": "पृष्ठताण म्हणजे द्रवाच्या पृष्ठभागावरील ताण. पान पस्तीस.", "language": "mr"}
    assert len(api.spoken) == 1                                 # nothing more was sent to the paid voice
    for _ in range(5):                                          # pressing play again and again is refused the same way...
        assert api.post("/api/speak", json=where).json()["message"].startswith("Today's voice budget")
    assert server.voice_limiter.room("testclient", 1)           # ...and uses up none of the visitor's own allowances
    assert server.visitor_voice.room("testclient", 1400)


def test_the_hourly_limit_has_its_own_message(api):  # noqa: F811
    for _ in range(2):
        reply = ask(api, text="प्रश्न")[0]
        api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]})
    reply = ask(api, text="प्रश्न")[0]
    capped = api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]})
    assert capped.json()["message"].startswith("You have reached this hour's voice limit")


# --- the access code ---------------------------------------------------------------------

def test_with_a_code_set_nothing_works_until_it_is_entered(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    assert api.get("/api/state").json() == {"locked": True}
    assert api.post("/api/ask", data={"text": "प्रश्न"}).status_code == 401
    assert api.asked == []
    assert api.get("/").status_code == 200                      # the page itself loads, to show the code screen
    assert api.post("/api/access", json={"code": "wrong"}).status_code == 403
    assert api.get("/api/state").json() == {"locked": True}
    assert api.post("/api/access", json={"code": "open-sesame"}).json() == {"ok": True}
    assert "voices" in api.get("/api/state").json()

    stranger = TestClient(server.app)
    stranger.cookies.set(server.ACCESS_COOKIE, api.cookies[server.ACCESS_COOKIE])
    assert stranger.get("/api/state").json() == {"locked": True}       # the pass is tied to the browser it was given to


def test_wrong_guesses_are_limited_and_right_entries_are_not_counted(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    monkeypatch.setattr(server, "code_limiter", server.RateLimiter(3, window_s=3600))
    for _ in range(5):                                          # a class of students entering the right code
        assert TestClient(server.app).post("/api/access", json={"code": "open-sesame"}).status_code == 200
    for _ in range(3):
        assert api.post("/api/access", json={"code": "guess"}).status_code == 403
    assert api.post("/api/access", json={"code": "open-sesame"}).status_code == 429     # after three wrong tries even the right one waits


# --- other hardening ---------------------------------------------------------------------

def test_uploads_are_limited_per_address(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "upload_limiter", server.RateLimiter(1, window_s=3600))
    monkeypatch.setattr(server, "ingest", lambda path, visitor_id, source=None: IngestResult(source or "sample.md", 1, 1, [], "mr"))
    assert api.post("/api/notes/sample").status_code == 200
    refused = api.post("/api/notes", files={"file": ("a.txt", b"text", "text/plain")})
    assert refused.status_code == 429
    assert "several files" in refused.json()["error"]
    assert TestClient(server.app).post("/api/notes/sample").status_code == 429      # a new cookie is still the same address


def test_every_response_carries_the_safety_headers_and_cookies_are_locked_down(api):  # noqa: F811
    opened = api.get("/api/state", headers={"x-forwarded-proto": "https"})
    policy = opened.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in policy                           # the page cannot be put inside another site
    assert "script-src 'self' https://cdn.jsdelivr.net/npm/katex@0.16.11/;" in policy      # scripts: its own and one library
    assert "object-src 'none'" in policy
    assert opened.headers["x-content-type-options"] == "nosniff"
    assert opened.headers["referrer-policy"] == "no-referrer"
    cookie = opened.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" in cookie                                           # over HTTPS only, when the site is HTTPS
    assert "secure" not in api.get("/api/state").headers["set-cookie"].lower()      # plain http on the laptop still works
    assert "frame-ancestors 'none'" in api.get("/").headers["content-security-policy"]
