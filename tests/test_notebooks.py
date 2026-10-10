"""Notebooks: each has its own sources, its own conversations and its own memory, and none sees another's."""
import psycopg
import pytest
from fastapi.testclient import TestClient

import server
from agent.overview import Overview
from rag.ingest import IngestResult
from tests.test_server import OUTLINE, api, ask  # noqa: F401  (api is a fixture)


def notebooks(client) -> list[dict]:
    return client.get("/api/state").json()["notebooks"]


def make(client, **body) -> str:
    made = client.post("/api/notebooks", json=body)
    assert made.status_code == 200
    return made.json()["notebook"]


# --- the list ------------------------------------------------------------------------

def test_a_new_browser_has_one_notebook_and_it_is_not_written_down_until_used(api):  # noqa: F811
    opened = api.get("/api/state").json()
    assert [n["title"] for n in opened["notebooks"]] == ["Untitled notebook"]
    assert opened["notebook"] == opened["notebooks"][0]["id"]
    assert api.shelf == {}                      # a browser that only looks costs the database nothing


def test_a_second_notebook_is_listed_and_both_are_saved(api):  # noqa: F811
    first = notebooks(api)[0]["id"]
    second = make(api)
    assert [n["id"] for n in notebooks(api)] == [first, second]
    assert list(api.shelf) == [first, second]   # the first one too, so their order survives a restart
    opened = api.get(f"/api/state?notebook={second}").json()
    assert (opened["notebook"], opened["notes"], opened["chats"], opened["chat"]) == (second, {"sources": []}, [], None)


def test_no_more_than_the_allowed_number_of_notebooks(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "MAX_NOTEBOOKS", 3)
    make(api)
    make(api)
    refused = api.post("/api/notebooks", json={})
    assert refused.status_code == 400
    assert "Delete one" in refused.json()["message"]
    assert len(notebooks(api)) == 3


def test_making_notebooks_over_and_over_is_limited_per_address(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "notebook_limiter", server.RateLimiter(1, window_s=3600))
    make(api)
    assert TestClient(server.app).post("/api/notebooks", json={}).status_code == 429     # a new cookie is the same address


# --- what is in one is not in another ----------------------------------------------------

def test_each_notebook_has_its_own_conversations_and_its_own_memory(api):  # noqa: F811
    first = notebooks(api)[0]["id"]
    second = make(api)
    in_first = ask(api, text="पहिल्या वहीतला प्रश्न", notebook=first)[0]["chat"]["id"]
    in_second = ask(api, text="दुसऱ्या वहीतला प्रश्न", notebook=second)[0]["chat"]["id"]
    asked_first, asked_second = api.asked[0]["session"], api.asked[1]["session"]
    assert (asked_first.id, asked_second.id) == (first, second)      # each searches only its own notes
    assert asked_first is not asked_second                           # and remembers only its own conversation
    assert [chat["id"] for chat in api.get(f"/api/state?notebook={first}").json()["chats"]] == [in_first]
    assert [chat["id"] for chat in api.get(f"/api/state?notebook={second}").json()["chats"]] == [in_second]
    # A conversation named from the other notebook is not continued here: a new one is started.
    assert ask(api, text="पुढचा प्रश्न", notebook=second, chat=in_first)[0]["chat"]["id"] not in (in_first, in_second)


def test_a_file_goes_into_the_notebook_it_was_added_to_and_nowhere_else(api, monkeypatch):  # noqa: F811
    stored = {}

    def fake_ingest(path, notebook_id, source=None):
        stored.setdefault(notebook_id, []).extend(OUTLINE)
        return IngestResult(source=source, pages=55, chunks=120, chapters=[1, 2], language="en")

    monkeypatch.setattr(server, "ingest", fake_ingest)
    monkeypatch.setattr(server, "notes_outline", lambda notebook_id: stored.get(notebook_id, []))
    first = notebooks(api)[0]["id"]
    second = make(api, title="Chemistry")
    added = api.post("/api/notes", files={"file": ("physics.pdf", b"%PDF", "application/pdf")}, data={"notebook": first})
    assert added.status_code == 200
    assert list(stored) == [first]
    assert api.get(f"/api/state?notebook={first}").json()["notes"]["sources"][0]["source"] == "physics.pdf"
    assert api.get(f"/api/state?notebook={second}").json()["notes"] == {"sources": []}
    ask(api, text="प्रश्न", notebook=first)
    ask(api, text="प्रश्न", notebook=second)
    assert api.asked[0]["session"].notes_languages == ["en"]
    assert api.asked[1]["session"].notes_languages == []        # the other notebook has no notes, in any language


def test_a_reply_in_any_notebook_can_be_spoken_and_rated(api):  # noqa: F811
    second = make(api)
    reply = ask(api, text="प्रश्न", notebook=second)[0]
    where = {"chat": reply["chat"]["id"], "key": reply["message"]["key"]}
    made = api.post("/api/speak", json=where)
    assert made.status_code == 200
    assert api.get(made.json()["url"]).status_code == 200
    assert api.post("/api/feedback", json={**where, "liked": True}).json() == {"saved": True}
    assert len(api.get(f"/api/chats/{where['chat']}").json()["messages"]) == 2


# --- names ---------------------------------------------------------------------------

def test_an_unnamed_notebook_takes_its_name_from_its_notes_and_a_given_name_is_kept(api):  # noqa: F811
    first = notebooks(api)[0]["id"]
    named = make(api, title="  My   revision\nnotes ")
    api.notes.extend(OUTLINE)
    api.get(f"/api/state?notebook={first}")
    api.get(f"/api/state?notebook={named}")
    assert [n["title"] for n in notebooks(api)] == ["Class 12 Physics", "My revision notes"]
    assert api.shelf[first][1] == "Class 12 Physics"            # saved under that name


def test_with_no_description_the_file_name_names_the_notebook(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "describe_notes", lambda notebook_id, language: Overview())
    api.notes.extend(OUTLINE)
    opened = api.get("/api/state").json()
    assert (opened["notebooks"][0]["title"], opened["notes"]["title"]) == ("physics", "physics")


def test_renaming_cleans_the_name_and_refuses_an_empty_one(api):  # noqa: F811
    first = notebooks(api)[0]["id"]
    renamed = api.post(f"/api/notebooks/{first}/rename", json={"title": " भौतिकशास्त्र\t१२ \x00\x1b " + "x" * 200})
    assert renamed.status_code == 200
    title = notebooks(api)[0]["title"]
    assert title.startswith("भौतिकशास्त्र १२ x")
    assert len(title) == server.MAX_TITLE_CHARS
    assert api.shelf[first][1] == title
    assert api.post(f"/api/notebooks/{first}/rename", json={"title": " \n "}).status_code == 400
    # Half a character, which no browser sends but a script can: dropped, not a crash.
    assert api.post(f"/api/notebooks/{first}/rename", content=b'{"title": "ok\\ud800"}', headers={"content-type": "application/json"}).status_code == 200
    assert notebooks(api)[0]["title"] == "ok"


# --- deleting -------------------------------------------------------------------------

def test_deleting_a_notebook_removes_what_it_held_and_leaves_the_others(api, monkeypatch, tmp_path):  # noqa: F811
    deleted, unlogged = [], []
    monkeypatch.setattr(server, "delete_session", deleted.append)
    monkeypatch.setattr(server, "forget_turns", unlogged.append)
    monkeypatch.setattr(server, "UPLOADS_DIR", tmp_path / "uploads")
    first = notebooks(api)[0]["id"]
    second = make(api)
    kept = ask(api, text="प्रश्न", notebook=first)[0]
    gone = ask(api, text="प्रश्न", notebook=second)[0]
    kept_audio = api.post("/api/speak", json={"chat": kept["chat"]["id"], "key": kept["message"]["key"]}).json()["url"]
    gone_audio = api.post("/api/speak", json={"chat": gone["chat"]["id"], "key": gone["message"]["key"]}).json()["url"]
    for notebook in (first, second):
        (tmp_path / "uploads" / notebook).mkdir(parents=True)
        (tmp_path / "uploads" / notebook / "book.pdf").write_bytes(b"%PDF")

    left = api.post(f"/api/notebooks/{second}/delete").json()
    assert left == {"notebook": first, "notebooks": [{"id": first, "title": "Untitled notebook"}]}
    assert deleted == unlogged == [second]                      # its notes and its logged turns, nobody else's
    assert not (tmp_path / "uploads" / second).exists()
    assert (tmp_path / "uploads" / first / "book.pdf").exists()
    assert list(api.shelf) == [first]
    assert api.get(gone_audio).status_code == 404
    assert api.get(kept_audio).status_code == 200
    assert api.get(f"/api/chats/{gone['chat']['id']}").status_code == 404
    assert api.post("/api/ask", data={"text": "प्रश्न", "notebook": second}).status_code == 404


def test_deleting_the_only_notebook_leaves_a_fresh_empty_one(api):  # noqa: F811
    only = notebooks(api)[0]["id"]
    ask(api, text="प्रश्न")
    left = api.post(f"/api/notebooks/{only}/delete").json()
    assert [n["title"] for n in left["notebooks"]] == ["Untitled notebook"]
    opened = api.get("/api/state").json()
    assert (opened["chats"], opened["notes"]) == ([], {"sources": []})


def test_remove_everything_deletes_every_notebook(api, monkeypatch):  # noqa: F811
    deleted = []
    monkeypatch.setattr(server, "delete_session", deleted.append)
    first = notebooks(api)[0]["id"]
    second = make(api)
    assert api.post("/api/forget").json() == {"ok": True}
    assert deleted == [first, second]
    assert api.shelf == {}
    assert notebooks(api)[0]["id"] not in (first, second)       # a new identity, with a new first notebook


# --- whose notebook it is ---------------------------------------------------------------

def test_a_notebook_can_only_be_used_by_the_browser_it_belongs_to(api):  # noqa: F811
    mine = make(api, title="Mine")
    ask(api, text="प्रश्न", notebook=mine)
    stranger = TestClient(server.app)
    assert stranger.post("/api/ask", data={"text": "प्रश्न", "notebook": mine}).status_code == 404
    assert stranger.post("/api/notes/sample", data={"notebook": mine}).status_code == 404
    assert stranger.post(f"/api/notebooks/{mine}/rename", json={"title": "Taken"}).status_code == 404
    assert stranger.post(f"/api/notebooks/{mine}/delete").status_code == 404
    seen = stranger.get(f"/api/state?notebook={mine}").json()       # opening the page with someone else's id: their own first notebook
    assert seen["notebook"] != mine
    assert seen["chats"] == []
    assert api.shelf[mine][1] == "Mine"
    assert len(api.asked) == 1


def test_an_id_that_is_not_one_is_refused_not_a_crash(api):  # noqa: F811
    for odd in ("..%2F..%2Fserver.py", "x" * 500, "%00", "'; DROP TABLE notebooks; --"):
        assert api.post(f"/api/notebooks/{odd}/delete").status_code == 404, odd
        assert api.post("/api/ask", data={"text": "प्रश्न", "notebook": odd}).status_code == 404, odd
    assert api.asked == []


def test_another_website_cannot_make_or_delete_notebooks(api):  # noqa: F811
    only = notebooks(api)[0]["id"]
    assert api.post("/api/notebooks", json={}, headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert api.post(f"/api/notebooks/{only}/delete", headers={"origin": "https://evil.example"}).status_code == 403
    assert len(notebooks(api)) == 1


# --- restarts and a database that is down --------------------------------------------------

def test_after_a_restart_the_notebooks_are_still_there_with_their_names(api, monkeypatch):  # noqa: F811
    first = notebooks(api)[0]["id"]
    second = make(api, title="Chemistry")
    ask(api, text="प्रश्न", notebook=second)
    monkeypatch.setattr(server, "notes_languages", lambda notebook_id: ["mr"] if notebook_id == second else [])
    server.visitors.clear()                                     # what a restart does to the app's memory
    opened = api.get(f"/api/state?notebook={second}").json()
    assert opened["notebooks"] == [{"id": first, "title": "Untitled notebook"}, {"id": second, "title": "Chemistry"}]
    assert opened["notebook"] == second
    assert opened["chats"] == []                                # conversations are kept in memory only
    ask(api, text="प्रश्न", notebook=second)
    assert api.asked[-1]["session"].notes_languages == ["mr"]   # and its notes are found again


def test_with_the_database_down_the_page_still_opens_and_notebooks_cannot_be_made(api, monkeypatch):  # noqa: F811
    def down(*_):
        raise psycopg.OperationalError("database is down")

    for name in ("stored_notebooks", "save_notebook", "remove_notebook", "notes_languages"):
        monkeypatch.setattr(server, name, down)
    assert len(notebooks(api)) == 1
    refused = api.post("/api/notebooks", json={})
    assert refused.status_code == 503
    assert "right now" in refused.json()["message"]
    assert len(notebooks(api)) == 1
    only = notebooks(api)[0]["id"]
    assert api.post(f"/api/notebooks/{only}/rename", json={"title": "Physics"}).status_code == 503
    assert notebooks(api)[0]["title"] == "Untitled notebook"   # the name on the page is the one that was saved


@pytest.mark.parametrize("busy", [True, False])
def test_the_app_starts_when_the_port_is_free_and_only_then(monkeypatch, busy):
    started = []
    monkeypatch.setattr(server, "_port_in_use", lambda port: busy)
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **options: started.append(options))
    if busy:
        with pytest.raises(SystemExit):
            server.main()
        assert started == []
    else:
        server.main()
        assert [options["port"] for options in started] == [server.PORT]
