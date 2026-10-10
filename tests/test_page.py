"""Reading a page of the book as a picture: finding the page, the reply it gives, and what happens when it cannot look."""
import httpx
import pytest

import agent.llm as llm_module
import agent.pipeline as pipeline
import agent.tools.page as page
from agent.errors import ServiceUnavailable
from agent.session import Session
from tests.conftest import USAGE, ScriptedLLM, fake_message


@pytest.fixture
def book(monkeypatch, tmp_path):
    """A stored book in which Example 1.6 is on printed page 12, the 21st page of the file."""
    seen = {"requests": [], "rendered": []}

    def fake_run(sql, params=()):
        if "ILIKE" in sql:
            return [("physics.pdf", 21, "12")] if "Example 1.6" in params[1] else []
        if "= %s ORDER BY id LIMIT 1" in sql:
            return [("physics.pdf", 21, "12")] if params[1] == "12" else []
        return [(f"text of page {params[2]}",)] if params[2] in (21, 22) else []

    def fake_pictures(pdf, positions, folder):
        seen["rendered"].append(positions)
        return [b"jpeg"] * len(positions)

    def fake_see(request, pictures):
        seen["requests"].append((request, len(pictures)))
        return "दिलेले: वस्तुमान 20 g.\n$$K = \\frac{1}{2} m v^2$$"

    stored = tmp_path / "physics.pdf"
    stored.write_bytes(b"%PDF")
    monkeypatch.setattr(page, "run", fake_run)
    monkeypatch.setattr(page, "_pictures", fake_pictures)
    monkeypatch.setattr(page, "see", fake_see)
    monkeypatch.setattr(page, "stored_file", lambda session_id, source: stored)
    seen["stored"] = stored
    return seen


def test_an_example_is_found_and_explained_from_the_picture_of_its_page(book):
    text, pages, final = page.open_page("s", None, "Example 1.6", "explain the solution", "mr")
    assert final
    assert text == "दिलेले: वस्तुमान 20 g.\n$$K = \\frac{1}{2} m v^2$$\nपान 12"       # the reading, then the page to cite
    assert pages == [{"source": "physics.pdf", "page": "12"}]
    assert book["rendered"] == [[21, 22]]                                              # the page and the one after it
    request, pictures = book["requests"][0]
    assert pictures == 2
    assert "Example 1.6: explain the solution" in request
    assert "Marathi" in request


def test_a_page_number_alone_is_enough(book):
    _text, pages, final = page.open_page("s", "12", None, "what is on this page", "en")
    assert final
    assert pages == [{"source": "physics.pdf", "page": "12"}]


def test_nothing_found_is_reported_to_the_model(book):
    assert page.open_page("s", "999", "Example 9.9", "explain", "mr") == (page.NOT_FOUND, [], False)


def test_when_the_page_cannot_be_looked_at_its_stored_text_is_used(book, monkeypatch):
    def busy(request, pictures):
        raise ServiceUnavailable("the page could not be read: RateLimitError")

    monkeypatch.setattr(page, "see", busy)
    text, pages, final = page.open_page("s", "12", None, "explain", "mr")
    assert not final
    assert text.startswith("The stored text of page 12 of physics.pdf (its equations may be scrambled):")
    assert "text of page 21\ntext of page 22" in text
    assert pages == [{"source": "physics.pdf", "page": "12"}]

    book["stored"].unlink()                             # the file is no longer kept: the same fallback, without trying to render
    book["rendered"].clear()
    assert page.open_page("s", "12", None, "explain", "mr")[2] is False
    assert book["rendered"] == []


# --- in the agent loop ---------------------------------------------------------------

@pytest.fixture
def loop(monkeypatch):
    monkeypatch.setattr(pipeline, "log_turn", lambda turn: 1)
    monkeypatch.setattr(pipeline, "log_speech", lambda turn: None)

    def install(*replies) -> ScriptedLLM:
        scripted = ScriptedLLM(*replies)
        monkeypatch.setattr(pipeline, "llm", scripted)
        return scripted

    return install


def test_the_reading_of_the_page_is_the_reply_and_is_remembered(loop, monkeypatch):
    asked = []
    reading = "Given: mass 20 g.\n$$K = \\frac{1}{2} m v^2$$\npage 12"
    monkeypatch.setattr(pipeline, "open_page", lambda *args: asked.append(args) or (reading, [{"source": "physics.pdf", "page": "12"}], True))
    scripted = loop(fake_message(tool="open_page", arguments='{"ask": "Explain the solution", "item": "Example 1.6", "page": 12}'))
    session = Session(id="s50")
    turn = pipeline.answer_turn(session, text="Can you explain example 1.6 on page 12, I did not understand the solution", hint="mr")
    assert asked == [("s50", "12", "Example 1.6", "Explain the solution", "en")]
    assert turn.reply_text == reading
    assert len(scripted.calls) == 1                         # no second model rewrites the working
    assert turn.sources == [{"source": "physics.pdf", "page": "12", "score": None}]
    assert session.messages[-1] == {"role": "assistant", "content": reading}       # a follow-up can build on it


def test_stored_text_is_explained_by_the_usual_model(loop, monkeypatch):
    monkeypatch.setattr(pipeline, "open_page", lambda *args: ("The stored text of page 12", [{"source": "n.md", "page": "12"}], False))
    scripted = loop(fake_message(tool="open_page", arguments='{"ask": "x", "page": 12}'), fake_message("Given: mass 20 g. page 12"))
    turn = pipeline.answer_turn(Session(id="s51"), text="explain what is on page 12 for me please", hint="mr")
    assert "what is given, what is asked" in scripted.calls[1][0][-1]["content"]
    assert turn.reply_text == "Given: mass 20 g. page 12"


# --- the main model is waited for when its limit is about to clear -------------------------

def limited(seconds: str) -> llm_module.groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return llm_module.groq.RateLimitError("rate limited", response=httpx.Response(429, request=request, headers={"retry-after": seconds}), body=None)


@pytest.mark.parametrize(("retry_after", "expected"), [("3", ["main", "main"]), ("30", ["main", "backup"])])
def test_a_short_rate_limit_on_the_main_model_is_waited_out(monkeypatch, retry_after, expected):
    used, waits = [], []

    def chat(model, messages, tools, max_tokens):
        used.append(model)
        if len(used) == 1:
            raise limited(retry_after)
        return fake_message("ok"), dict(USAGE)

    monkeypatch.setattr(llm_module, "LLM_MODEL", "main")
    monkeypatch.setattr(llm_module, "LLM_FALLBACK_MODEL", "backup")
    monkeypatch.setattr(llm_module, "_chat", chat)
    monkeypatch.setattr(llm_module.time, "sleep", waits.append)
    llm_module.llm([{"role": "user", "content": "hi"}])
    assert used == expected
    assert waits == ([3.0] if expected[1] == "main" else [])


# --- a named page or example goes to the page reader, whatever the model searched for ----------

@pytest.mark.parametrize(("text", "expected"), [
    ("example 1.6 formulas explanation", {"page": None, "item": "Example 1.6"}),
    ("पान 12 वरचं example 1.6 मध्ये प्रश्न काय विचारला आहे", {"page": 12, "item": "Example 1.6"}),
    ("Explain the example 1.6 on page number 12", {"page": 12, "item": "Example 1.6"}),
    ("what does figure 2.3 show", {"page": None, "item": "Fig. 2.3"}),
    ("उदाहरण 4.2 समजावून सांग", {"page": None, "item": "Example 4.2"}),
    ("पेज 45 वर काय आहे", {"page": 45, "item": None}),
    ("surface tension definition", None),
    ("explain with an example", None),
    ("the value is 9.8 newton", None),
])  # fmt: skip
def test_named_reference(text, expected):
    assert pipeline.named_reference(text) == expected


def test_a_search_for_a_named_example_is_turned_into_reading_its_page(loop, monkeypatch):
    asked = []
    monkeypatch.setattr(pipeline, "search_notes", lambda *_: pytest.fail("a search cannot explain a worked example"))
    monkeypatch.setattr(pipeline, "open_page", lambda *args: asked.append(args) or ("The formulas are…\npage 12", [{"source": "physics.pdf", "page": "12"}], True))
    loop(fake_message(tool="search_notes", arguments='{"query": "example 1.6 formulas explanation", "kind": "fact"}'))
    turn = pipeline.answer_turn(Session(id="s52"), text="Can you explain all the formulas we did in this example in detail", hint="mr")
    assert asked == [("s52", None, "Example 1.6", "Can you explain all the formulas we did in this example in detail", "en")]
    assert turn.tool_name == "open_page"
    assert turn.reply_text == "The formulas are…\npage 12"
