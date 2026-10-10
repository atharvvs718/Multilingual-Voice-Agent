"""What an attacker might try, and what the app does about it. Each test is one finding from the security review."""
import hashlib
import hmac
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from fastapi.testclient import TestClient
from starlette.requests import Request

import agent.audio as audio
import agent.sarvam as sarvam_api
import server
from agent import budget
from agent.errors import BadClip, ServiceUnavailable
from agent.limits import RateLimiter
from agent.turn import Turn
from rag.ingest import find_chapter, printed_page_offset
from tests.conftest import memory_budget
from tests.test_server import OUTLINE, api, ask  # noqa: F401  (api is a fixture)

# --- getting in without the code, or past the limits ---------------------------------------------

def test_a_crafted_host_header_does_not_get_past_the_access_code(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    for host in ("x?", "x#", "x/..", "evil.example"):
        refused = api.post("/api/ask", data={"text": "प्रश्न"}, headers={"host": host})
        assert refused.status_code == 401, host
    assert api.asked == []


def test_a_malformed_access_cookie_is_refused_not_a_crash(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    garbled = "access=caf\u00e9".encode("latin-1")      # bytes a browser would never send, but an attacker can
    assert api.get("/api/state", headers={"cookie": garbled}).json() == {"locked": True}


def test_a_made_up_forwarded_address_does_not_give_a_fresh_allowance(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "limiter", RateLimiter(1, window_s=3600))
    ask(api, text="पहिला प्रश्न")
    response = api.post("/api/ask", data={"text": "दुसरा प्रश्न"}, headers={"x-forwarded-for": "9.9.9.9"})
    assert "मर्यादा" in response.text                    # still the same caller: the header is ignored with no proxy in front
    assert len(api.asked) == 1


def asking_from(*forwarded: str) -> Request:
    """A request arriving from 10.0.0.1 with these X-Forwarded-For header lines."""
    headers = [(b"x-forwarded-for", line.encode()) for line in forwarded]
    return Request({"type": "http", "headers": headers, "client": ("10.0.0.1", 50000)})


def test_behind_one_proxy_only_the_address_the_proxy_wrote_is_used(monkeypatch):
    monkeypatch.setattr(server, "PROXY_HOPS", 1)
    assert server._address(asking_from("1.2.3.4, 203.0.113.7")) == "203.0.113.7"   # the first entry is the sender's own claim
    assert server._address(asking_from("6.6.6.6", "203.0.113.7")) == "203.0.113.7"    # also when sent as two header lines
    assert server._address(asking_from()) == "10.0.0.1"
    monkeypatch.setattr(server, "PROXY_HOPS", 0)
    assert server._address(asking_from("1.2.3.4")) == "10.0.0.1"


def test_a_pass_cannot_be_made_from_a_guess_at_the_code(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    visitor = "0" * 32
    forged = hmac.new(b"open-sesame", visitor.encode(), hashlib.sha256).hexdigest()      # how the pass used to be made
    stranger = TestClient(server.app)
    stranger.cookies.set(server.COOKIE, visitor)
    stranger.cookies.set(server.ACCESS_COOKIE, forged)
    assert stranger.get("/api/state").json() == {"locked": True}       # even with the right code: only /api/access gives a pass


def test_odd_input_to_the_gate_is_refused_not_a_crash(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "ACCESS_CODE", "open-sesame")
    assert api.post("/api/access", content=b'{"code": "\\ud800"}', headers={"content-type": "application/json"}).status_code == 403
    assert api.get("/api/state", headers={"host": "["}).json() == {"locked": True}


def test_another_website_cannot_act_as_the_visitor(api, monkeypatch):  # noqa: F811
    deleted = []
    monkeypatch.setattr(server, "delete_session", deleted.append)
    ask(api, text="प्रश्न")
    before = api.cookies[server.COOKIE]
    for headers in ({"sec-fetch-site": "cross-site"}, {"sec-fetch-site": "same-site"}, {"origin": "http://localhost:3000"}):
        refused = api.post("/api/forget", headers=headers)
        assert refused.status_code == 403, headers
        assert "set-cookie" not in refused.headers                      # and it is not handed a new id either
    assert deleted == []
    assert api.cookies[server.COOKIE] == before
    assert api.post("/api/forget", headers={"sec-fetch-site": "same-origin"}).status_code == 200


def test_a_body_that_never_says_how_long_it_is_is_cut_off(api):  # noqa: F811
    def endless():
        for _ in range(40):
            yield b"x" * 4096           # 160 kB, sent in pieces with no Content-Length

    read = []
    real_json = server.Request.json

    async def watched_json(request):
        body = await real_json(request)
        read.append(body)
        return body

    server.Request.json = watched_json
    try:
        refused = api.post("/api/access", content=endless(), headers={"content-type": "application/json"})
        declared = api.post("/api/access", json={"code": "x" * 100_000})
    finally:
        server.Request.json = real_json
    assert refused.status_code in (400, 413)        # refused either way; which code depends on who was reading
    assert declared.status_code in (400, 413)
    assert read == []                               # and in neither case was the whole body taken in


def test_only_the_pages_own_two_files_are_served(api):  # noqa: F811
    assert api.get("/static/app.js").status_code == 200
    assert api.get("/static/style.css").headers["content-type"].startswith("text/css")
    for name in ("index.html", "%5C%5Chost%5Cshare%5Cx", "..%2Fserver.py", "..%5Cconfig.py"):
        assert api.get(f"/static/{name}").status_code == 404, name


def test_the_api_description_is_not_published(api):  # noqa: F811
    assert api.get("/openapi.json").status_code == 404
    assert api.get("/docs").status_code == 404


# --- what a visitor is shown, and what is stored ---------------------------------------------------

def test_error_details_shown_to_a_visitor_name_the_failure_only(api, monkeypatch):  # noqa: F811
    def failing(session, **_):
        return Turn(session_id=session.id, transcript="प्रश्न", reply_text="क्षमस्व, पुन्हा प्रयत्न करा.", id=1, language="mr",
                    error='OperationalError: connection failed: host=db.internal user=owner password=hunter2')

    monkeypatch.setattr(server, "answer_turn", failing)
    details = ask(api, text="प्रश्न")[0]["details"]
    assert details["error"] == "OperationalError"
    assert "hunter2" not in str(details)
    assert "db.internal" not in str(details)


def test_notes_are_stored_under_a_hash_of_the_cookie_not_the_cookie(api):  # noqa: F811
    ask(api, text="प्रश्न")
    cookie = api.cookies[server.COOKIE]
    stored_as = api.asked[0]["session"].id
    assert stored_as != cookie
    assert stored_as == hashlib.sha256(cookie.encode()).hexdigest()[:32]


def test_remove_my_notes_deletes_everything_and_gives_a_new_identity(api, monkeypatch, tmp_path):  # noqa: F811
    deleted, unlogged = [], []
    monkeypatch.setattr(server, "delete_session", deleted.append)
    monkeypatch.setattr(server, "forget_turns", unlogged.append)
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path / "uploads")
    reply = ask(api, text="प्रश्न")[0]
    made = api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]}).json()["url"]
    before = api.cookies[server.COOKIE]
    folder = tmp_path / "uploads" / api.asked[0]["session"].id
    folder.mkdir(parents=True)
    (folder / "book.pdf").write_bytes(b"%PDF")

    assert api.post("/api/forget").json() == {"ok": True}
    assert deleted == unlogged == [api.asked[0]["session"].id]     # the notes, and the logged questions and replies
    assert not folder.exists()
    assert not (tmp_path / made.rsplit("/", 1)[1]).exists()         # the spoken reply is gone too
    assert api.cookies[server.COOKIE] != before
    assert api.get("/api/state").json()["chats"] == []


def test_switching_language_back_and_forth_describes_the_notes_once_each(api, monkeypatch):  # noqa: F811
    api.notes.extend(OUTLINE)
    calls = []
    monkeypatch.setattr(server, "describe_notes", lambda visitor_id, language: calls.append(language) or server.Overview("T", "S", ["Q?"]))
    for language in ("mr", "hi", "mr", "hi", "en", "mr"):
        api.get(f"/api/state?language={language}")
    assert calls == ["mr", "hi", "en"]


# --- input that is too big or built to be slow --------------------------------------------------------

def test_a_very_long_question_is_refused_before_any_model_sees_it(api):  # noqa: F811
    refused = api.post("/api/ask", data={"text": "x " * 600})
    assert refused.status_code == 400
    assert api.asked == []
    assert not [chat for visitor in server.visitors.values() for chat in visitor.chats()]       # not even an empty conversation was kept


def test_a_chapter_number_that_is_not_one_is_ignored(api):  # noqa: F811
    assert ask(api, action="quiz", chapter="100")[0]["text"] == "मला एक प्रश्न विचार"
    assert ask(api, action="quiz", chapter="12; DROP")[0]["text"] == "मला एक प्रश्न विचार"


def test_the_conversation_in_use_is_not_the_one_dropped(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "MAX_CHATS", 2)
    monkeypatch.setattr(server, "limiter", RateLimiter(100, window_s=3600))
    first = ask(api, text="पहिला")[0]["chat"]["id"]
    second = ask(api, text="दुसरा")[0]["chat"]["id"]
    ask(api, text="पहिल्यावर पुढे", chat=first)                  # the first is used again, so the second is now the oldest
    ask(api, text="तिसरा")
    assert api.get(f"/api/chats/{first}").status_code == 200
    assert api.get(f"/api/chats/{second}").status_code == 404


def test_many_page_loads_at_once_describe_the_notes_once(api, monkeypatch):  # noqa: F811
    api.notes.extend(OUTLINE)
    calls = []

    def slow_describe(visitor_id, language):
        calls.append(language)
        time.sleep(0.05)
        return server.Overview("T", "S", ["Q?"])

    monkeypatch.setattr(server, "describe_notes", slow_describe)
    api.get("/api/state")                                       # sets the cookie
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda _: api.get("/api/state?language=hi"), range(8)))
    assert calls == ["mr", "hi"]


def test_past_the_hourly_allowance_notes_go_undescribed(api, monkeypatch):  # noqa: F811
    api.notes.extend(OUTLINE)
    monkeypatch.setattr(server, "describe_limiter", RateLimiter(1, window_s=3600))
    monkeypatch.setattr(server, "describe_notes", lambda visitor_id, language: server.Overview("Described", "S", ["Q?"]))
    assert api.get("/api/state").json()["notes"]["title"] == "Described"
    again = api.get("/api/state?language=hi").json()["notes"]
    assert (again["summary"], again["questions"]) == ("", [])       # no further call was made for the second language
    assert again["title"] == "Described"                            # the notebook keeps the name it has


def test_the_decoder_is_told_to_stop_and_what_it_says_stays_in_the_log(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        return subprocess.CompletedProcess(cmd, 1, stdout=b"", stderr=b"C:\\Users\\someone\\Temp\\clip.webm: Invalid data")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    with pytest.raises(BadClip) as refused:
        audio.load_16k_mono("clip.webm")
    assert "someone" not in str(refused.value)
    assert seen["cmd"][seen["cmd"].index("-t") + 1] == str(audio.MAX_CLIP_S + 5)
    assert seen["kwargs"]["timeout"] == audio.DECODE_TIMEOUT_S
    assert not any("KEY" in name for name in seen["kwargs"]["env"])       # the helper is started without the secrets


def test_a_file_of_blank_lines_is_read_in_no_time():
    blank = "\n" * 200_000                  # with nothing after them: the case that took minutes
    started = time.perf_counter()
    assert find_chapter(blank, current=1) is None
    assert find_chapter(blank + "2.1 Introduction: text", current=1) == 2
    assert printed_page_offset([(index, blank) for index in range(1, 4)]) is None
    assert time.perf_counter() - started < 2


def test_headings_in_the_forms_real_books_use_are_still_found():
    assert find_chapter("CHAPTER\r\n6\r\nLight") == 6
    assert find_chapter("धडा\r\n६") == 6
    assert find_chapter("\x0cChapter\u00a06") == 6                       # a form feed, and a non-breaking space
    assert find_chapter("Can you recall?\n2.1\r\nIntroduction to fluids\n2.2 Fluid: text", current=1) == 2
    assert find_chapter("\u00a02.1 Introduction: text\n 2.2\u00a0Fluid: text", current=1) == 2


def test_a_pages_own_number_is_found_once_and_never_invented():
    assert printed_page_offset([(index, f"{index - 2}\r\n") for index in range(3, 9)]) == 2       # pages that are only their number
    assert printed_page_offset([(index, "42" + " " * 198 + "Chapter 3 Motion") for index in range(1, 9)]) is None
    long_page = "text " * 100
    assert printed_page_offset([(index, f"{long_page}\n{index - 4}\n") for index in range(5, 12)]) == 4


# --- money ------------------------------------------------------------------------------------------

def test_the_voice_is_charged_for_the_text_that_is_really_sent(api, monkeypatch):  # noqa: F811
    digits = "7777 " * 20                           # 100 characters as written, far more once spelt out for speech
    monkeypatch.setattr(server, "answer_turn", lambda session, **_: Turn(session_id=session.id, transcript="q", reply_text=digits, id=1, language="mr"))
    monkeypatch.setattr(budget, "daily", memory_budget(1.0))
    reply = ask(api, text="प्रश्न")[0]
    refused = api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]})
    assert refused.status_code == 429               # Rs 0.30 as written would have fitted; as sent it does not
    assert api.spoken == []


def test_one_visitor_cannot_use_the_whole_days_voice(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "voice_limiter", RateLimiter(100, window_s=3600))
    monkeypatch.setattr(server, "visitor_voice", RateLimiter(20, window_s=24 * 3600))       # 20 paise for this visitor
    first = ask(api, text="प्रश्न")[0]
    assert api.post("/api/speak", json={"chat": first["chat"]["id"], "key": first["message"]["key"]}).status_code == 200
    second = ask(api, text="प्रश्न")[0]
    capped = api.post("/api/speak", json={"chat": second["chat"]["id"], "key": second["message"]["key"]})
    assert capped.status_code == 429
    assert capped.json()["message"].startswith("You have used your share of today's voice")
    assert budget.daily.spent() < 1                 # the day's budget is far from used: it was this visitor's share


def test_a_paid_request_that_timed_out_is_not_sent_again(monkeypatch):
    sent = []

    def slow(url, **_):
        sent.append(url)
        raise requests.ReadTimeout("no answer in time")

    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    monkeypatch.setattr(sarvam_api.requests, "post", slow)
    with pytest.raises(ServiceUnavailable, match="ReadTimeout"):
        sarvam_api.post("text-to-speech", json={"text": "नमस्ते"})
    assert len(sent) == 1


@pytest.mark.parametrize("failure", [requests.ConnectionError("connection aborted"), requests.exceptions.ChunkedEncodingError("cut off")])
def test_a_paid_request_that_broke_after_it_was_sent_is_not_sent_again(monkeypatch, failure):
    sent = []

    def broken(url, **_):
        sent.append(url)
        raise failure

    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    monkeypatch.setattr(sarvam_api.requests, "post", broken)
    with pytest.raises(ServiceUnavailable):
        sarvam_api.post("text-to-speech", json={"text": "नमस्ते"})
    assert len(sent) == 1


def test_only_a_request_sarvam_did_not_take_is_sent_again(monkeypatch):
    class Answer:
        def __init__(self, status):
            self.status_code, self.ok, self.text = status, status < 400, ""

        def json(self):
            return {"done": True}

    answers = [Answer(503), Answer(429), Answer(500)]
    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    monkeypatch.setattr(sarvam_api.requests, "post", lambda url, **_: answers.pop(0))
    with pytest.raises(ServiceUnavailable, match="HTTP 500"):
        sarvam_api.post("text-to-speech", json={"text": "नमस्ते"})
    assert answers == []                            # busy twice, so sent a third time; the 500 ended it
