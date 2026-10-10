"""The API the page talks to, with the pipeline, the notes store and the voice replaced by fakes."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server
from agent.errors import BadClip
from agent.limits import RateLimiter
from agent.overview import Overview
from agent.turn import Turn
from rag.ingest import IngestError, IngestResult

OUTLINE = [
    {"source": "physics.pdf", "chapter": 1, "title": "Rotational Dynamics", "first_page": "1", "last_page": "25", "pages": 25},
    {"source": "physics.pdf", "chapter": 2, "title": "Fluids", "first_page": "26", "last_page": "55", "pages": 30},
]


@pytest.fixture
def api(monkeypatch, tmp_path):
    """A browser talking to the app. api.asked lists what reached the pipeline; api.spoken what reached the voice."""
    asked, spoken, notes = [], [], []
    shelf = {}      # the notebooks table: id -> (owner, title), in the order they were saved

    def fake_answer(session, *, audio_path=None, text=None, hint=None, on_heard=None):
        asked.append({"text": text, "hint": hint, "clip": audio_path, "clip_existed": bool(audio_path) and Path(audio_path).exists(),
                      "session": session})  # fmt: skip
        if audio_path and Path(audio_path).read_bytes() == b"silence":
            raise BadClip("No speech heard. Please try again.")
        if on_heard and audio_path:
            on_heard("सरफेस टेन्शन म्हणजे काय?")
        return Turn(
            session_id=session.id, transcript=text or "सरफेस टेन्शन म्हणजे काय?", language="mr", id=len(asked),
            reply_text="पृष्ठताण म्हणजे द्रवाच्या पृष्ठभागावरील ताण. पान 35.", tool_name="search_notes",
            tool_args={"query": "surface tension definition"}, sources=[{"source": "physics.pdf", "page": "35", "score": 0.9}],
        )  # fmt: skip

    def fake_speak(turn, voice):
        spoken.append(voice)
        path = tmp_path / f"reply_{len(spoken):08x}.wav"
        path.write_bytes(b"RIFF audio")
        return str(path)

    monkeypatch.setattr(server, "answer_turn", fake_answer)
    monkeypatch.setattr(server, "speak", fake_speak)
    monkeypatch.setattr(server, "OUT_DIR", tmp_path)
    monkeypatch.setattr(server, "stored_file", lambda visitor_id, name: tmp_path / "uploads" / visitor_id / name)
    monkeypatch.setattr(server, "notes_outline", lambda visitor_id: list(notes))
    monkeypatch.setattr(server, "notes_languages", lambda visitor_id: [])
    monkeypatch.setattr(server, "describe_notes", lambda visitor_id, language: Overview("Class 12 Physics", "Two sentences.", ["कृष्णपिंड म्हणजे काय?"]))
    monkeypatch.setattr(server, "set_feedback", lambda turn_id, value: True)
    monkeypatch.setattr(server, "limiter", RateLimiter(30, window_s=3600))
    monkeypatch.setattr(server, "voice_limiter", RateLimiter(2, window_s=3600))
    monkeypatch.setattr(server, "upload_limiter", RateLimiter(6, window_s=3600))
    monkeypatch.setattr(server, "visitor_voice", RateLimiter(1500, window_s=24 * 3600))
    monkeypatch.setattr(server, "describe_limiter", RateLimiter(20, window_s=3600))
    monkeypatch.setattr(server, "code_limiter", RateLimiter(10, window_s=3600))
    monkeypatch.setattr(server, "forget_turns", lambda session_id: None)
    monkeypatch.setattr(server, "delete_session", lambda session_id: None)
    monkeypatch.setattr(server, "stored_notebooks", lambda owner: [(key, title) for key, (whose, title) in shelf.items() if whose == owner])
    monkeypatch.setattr(server, "save_notebook", lambda key, owner, title: shelf.__setitem__(key, (owner, title)))
    monkeypatch.setattr(server, "remove_notebook", lambda key, owner: shelf.pop(key, None))
    monkeypatch.setattr(server, "forget_notebooks", lambda owner: [shelf.pop(key) for key in [k for k, v in shelf.items() if v[0] == owner]])
    monkeypatch.setattr(server, "notebook_limiter", RateLimiter(30, window_s=3600))
    monkeypatch.setattr(server, "visitors", {})
    client = TestClient(server.app)
    client.asked, client.spoken, client.notes, client.shelf = asked, spoken, notes, shelf
    return client


def ask(client, **fields) -> list[dict]:
    """Send one question and return the events that came back."""
    files = {"audio": ("question", fields.pop("audio"), fields.pop("kind", "audio/webm;codecs=opus"))} if "audio" in fields else None
    response = client.post("/api/ask", data={"language": "mr", **fields}, files=files)
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines()]


# --- opening the page --------------------------------------------------------------

def test_a_new_browser_gets_an_id_the_voices_and_an_empty_page(api):
    opened = api.get("/api/state")
    body = opened.json()
    assert server._VISITOR_ID.fullmatch(opened.cookies[server.COOKIE])
    assert [voice["id"] for voice in body["voices"]] == ["priya", "pooja", "neha", "shubh", "aditya"]
    assert body["voice"] == "priya"
    assert body["notes"] == {"sources": []}
    assert body["chats"] == []
    assert body["chat"] is None
    assert api.get("/").status_code == 200                 # the page itself
    assert api.get("/static/app.js").status_code == 200


def test_notes_are_described_once_and_shown_with_their_chapters(api, monkeypatch):
    api.notes.extend(OUTLINE)
    calls = []
    monkeypatch.setattr(server, "describe_notes", lambda visitor_id, language: calls.append(language) or Overview("Class 12 Physics", "S.", ["Q?"]))
    notes = api.get("/api/state").json()["notes"]
    assert notes["sources"] == [{"source": "physics.pdf", "pages": 55, "chapters": [1, 2]}]
    assert (notes["title"], notes["chapters"], notes["pages"], notes["questions"]) == ("Class 12 Physics", [1, 2], 55, ["Q?"])
    api.get("/api/state")
    assert calls == ["mr"]                                 # the same notes in the same language are not described again
    api.get("/api/state?language=hi")
    assert calls == ["mr", "hi"]                           # the questions to try follow the language spoken


# --- asking ------------------------------------------------------------------------

def test_a_typed_question_is_answered_kept_and_named(api):
    events = ask(api, text="  Surface tension म्हणजे काय?  ")
    assert [event["event"] for event in events] == ["reply"]
    reply = events[0]
    assert reply["message"]["text"].startswith("पृष्ठताण")
    assert reply["message"]["cited"] == [{"source": "physics.pdf", "pages": ["35"]}]
    assert reply["speak"] is False                          # typed: spoken only when the student asks
    assert reply["chat"]["title"] == "Surface tension definition"
    assert api.asked[0]["text"] == "Surface tension म्हणजे काय?"
    assert api.asked[0]["hint"] == "mr"

    body = api.get("/api/state").json()
    assert body["chats"] == [reply["chat"]]
    assert [(m["role"], m["spoken"]) for m in body["chat"]["messages"]] == [("user", False), ("assistant", False)]


def test_a_spoken_question_is_shown_as_soon_as_it_is_heard_and_the_clip_is_not_kept(api):
    events = ask(api, audio=b"a recording")
    assert [event["event"] for event in events] == ["heard", "reply"]
    assert events[0]["text"] == "सरफेस टेन्शन म्हणजे काय?"
    assert events[1]["speak"] is True                       # a spoken question is answered aloud
    assert api.asked[0]["clip_existed"]
    assert api.asked[0]["text"] is None
    assert not Path(api.asked[0]["clip"]).exists()
    assert api.get("/api/state").json()["chat"]["messages"][0]["spoken"] is True


def test_a_clip_with_no_speech_gives_a_message_and_keeps_nothing(api):
    events = ask(api, audio=b"silence")
    assert events == [{"event": "error", "message": "No speech heard. Please try again."}]
    assert api.get("/api/state").json()["chats"] == []
    assert not Path(api.asked[0]["clip"]).exists()


def test_an_upload_that_is_not_audio_never_reaches_the_pipeline(api):
    events = ask(api, audio=b"<html>", kind="text/html")
    assert events[0]["event"] == "error"
    assert api.asked == []


def test_follow_ups_stay_in_their_conversation_and_new_chats_get_their_own_memory(api):
    first = ask(api, text="पहिला प्रश्न")[0]["chat"]["id"]
    assert ask(api, text="दुसरा प्रश्न", chat=first)[0]["chat"]["id"] == first
    second = ask(api, text="नवीन विषय")[0]["chat"]["id"]
    assert second != first
    assert api.asked[0]["session"] is api.asked[1]["session"]
    assert api.asked[2]["session"] is not api.asked[0]["session"]
    assert api.asked[2]["session"].id == api.asked[0]["session"].id       # the notes are the notebook's, shared by its chats
    assert [chat["id"] for chat in api.get("/api/state").json()["chats"]] == [second, first]      # newest first
    assert len(api.get(f"/api/chats/{first}").json()["messages"]) == 4
    assert api.get("/api/chats/nothing").status_code == 404


def test_the_study_tools_ask_in_the_students_language(api):
    events = ask(api, action="quiz", chapter="12")
    assert events[0] == {"event": "heard", "text": "धडा 12 वर मला एक प्रश्न विचार"}
    assert api.asked[0]["text"] == "धडा 12 वर मला एक प्रश्न विचार"
    ask(api, action="quiz", language="hi")
    assert api.asked[1]["text"] == "मुझसे एक प्रश्न पूछो"
    ask(api, action="summary", chapter="8")
    assert api.asked[2]["text"] == "धडा 8 चा सारांश सांग"
    ask(api, action="summary")
    assert api.asked[3]["text"] == "संपूर्ण पुस्तकाचा सारांश सांग"
    events = ask(api, action="summary", chapter="8", language="en")
    assert events[0]["text"] == "Summarise chapter 8 for me"
    assert api.asked[4]["hint"] == "en"                     # English can be chosen as the language spoken


def test_over_the_hourly_cap_the_student_is_told_and_nothing_is_asked(api, monkeypatch):
    monkeypatch.setattr(server, "limiter", RateLimiter(1, window_s=3600))
    ask(api, text="पहिला प्रश्न")
    events = ask(api, text="दुसरा प्रश्न")
    assert events[0]["message"]["text"] == "तुम्ही या तासाची मर्यादा गाठली आहे. थोड्या वेळाने पुन्हा या."
    assert events[0]["message"]["can_speak"] is False
    assert len(api.asked) == 1


def test_asking_nothing_is_refused(api):
    assert api.post("/api/ask", data={"text": "   "}).status_code == 400


# --- the voice ---------------------------------------------------------------------

def test_a_reply_is_spoken_once_per_voice_and_only_its_owner_can_fetch_it(api):
    reply = ask(api, text="प्रश्न")[0]
    where = {"chat": reply["chat"]["id"], "key": reply["message"]["key"]}
    made = api.post("/api/speak", json={**where, "voice": "pooja"}).json()
    assert api.spoken == ["pooja"]
    assert api.get(made["url"]).content == b"RIFF audio"
    assert api.post("/api/speak", json={**where, "voice": "pooja"}).json()["url"] == made["url"]
    assert api.spoken == ["pooja"]                          # the same voice again: the same file, nothing new made
    api.post("/api/speak", json={**where, "voice": "nobody"})
    assert api.spoken == ["pooja", "priya"]                 # an unknown voice falls back to the default

    stranger = TestClient(server.app)
    assert stranger.get(made["url"]).status_code == 404
    assert stranger.post("/api/speak", json={**where, "voice": "pooja"}).status_code == 404
    assert api.get("/api/audio/..%2Fconfig.py").status_code == 404


def test_over_the_voice_cap_the_reply_is_not_spoken(api):
    for _ in range(2):
        reply = ask(api, text="प्रश्न")[0]
        assert api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]}).status_code == 200
    reply = ask(api, text="प्रश्न")[0]
    capped = api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]})
    assert capped.status_code == 429
    assert "voice limit" in capped.json()["message"]
    assert len(api.spoken) == 2


def test_a_voice_that_cannot_be_made_is_reported(api, monkeypatch):
    monkeypatch.setattr(server, "speak", lambda turn, voice: None)
    reply = ask(api, text="प्रश्न")[0]
    failed = api.post("/api/speak", json={"chat": reply["chat"]["id"], "key": reply["message"]["key"]})
    assert failed.status_code == 503
    assert "could not be made" in failed.json()["message"]


def test_feedback_is_saved_with_the_reply(api):
    reply = ask(api, text="प्रश्न")[0]
    where = {"chat": reply["chat"]["id"], "key": reply["message"]["key"]}
    assert api.post("/api/feedback", json={**where, "liked": False}).json() == {"saved": True}
    assert api.get(f"/api/chats/{where['chat']}").json()["messages"][1]["liked"] is False


# --- notes -------------------------------------------------------------------------

def test_an_uploaded_file_is_indexed_under_its_own_name_and_then_removed(api, monkeypatch):
    seen = {}

    def fake_ingest(path, visitor_id, source=None):
        seen.update(path=Path(path), text=Path(path).read_text(), source=source)
        api.notes.extend(OUTLINE)
        return IngestResult(source=source, pages=55, chunks=120, chapters=[1, 2], language="en")

    monkeypatch.setattr(server, "ingest", fake_ingest)
    added = api.post("/api/notes", files={"file": ("../my physics.PDF", b"the book", "application/pdf")}, data={"language": "mr"})
    assert added.status_code == 200
    assert added.json()["status"] == "Added my physics.PDF."
    assert added.json()["notes"]["title"] == "Class 12 Physics"
    assert (seen["text"], seen["source"], seen["path"].suffix) == ("the book", "my physics.PDF", ".pdf")
    assert not seen["path"].exists()
    ask(api, text="प्रश्न")
    assert api.asked[0]["session"].notes_languages == ["en"]       # every conversation knows what language the notes are in


def test_a_file_that_cannot_be_indexed_says_why(api, monkeypatch):
    def refuse(path, visitor_id, source=None):
        raise IngestError("Only .pdf, .txt and .md files are supported.")

    monkeypatch.setattr(server, "ingest", refuse)
    refused = api.post("/api/notes", files={"file": ("notes.docx", b"x", "application/octet-stream")})
    assert refused.status_code == 400
    assert refused.json() == {"error": "Only .pdf, .txt and .md files are supported."}


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js to parse the script")
def test_the_page_script_parses():
    # One wrong character here leaves the whole page dead, and no test of the server would notice.
    checked = subprocess.run(["node", "--check", str(server.WEB / "app.js")], capture_output=True, text=True, timeout=30)
    assert checked.returncode == 0, checked.stderr[:500]
