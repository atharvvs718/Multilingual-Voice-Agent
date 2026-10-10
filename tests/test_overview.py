"""The notes guide: parsing the model's reply, and failing quietly."""
import agent.overview as overview_module
from agent.errors import ServiceUnavailable
from agent.overview import Overview, describe_notes, parse
from tests.conftest import USAGE, fake_message

REPLY = """TITLE: Class 12 Physics textbook
SUMMARY: The notes cover rotational dynamics and fluids. They include laws and worked examples.
Q: केंद्राभिमुख बल म्हणजे काय?
Q: पृष्ठताण कसा मोजतात?
Q: बर्नोलीचे समीकरण काय सांगते?
Q: a fourth question that should be ignored"""


def test_parse_reads_title_summary_and_three_questions():
    guide = parse(REPLY)
    assert guide.title == "Class 12 Physics textbook"
    assert guide.summary.startswith("The notes cover rotational dynamics")
    assert guide.questions == ["केंद्राभिमुख बल म्हणजे काय?", "पृष्ठताण कसा मोजतात?", "बर्नोलीचे समीकरण काय सांगते?"]


def test_parse_copes_with_a_reply_in_the_wrong_shape():
    assert parse("Sorry, I cannot help with that.") == Overview()


def test_describe_notes_uses_the_sample_and_the_student_language(monkeypatch):
    sent = []

    def fake_llm(messages, **_):
        sent.append(messages)
        return fake_message(REPLY), USAGE

    monkeypatch.setattr(overview_module, "_sample", lambda session_id: "Files: physics.pdf\n\nCircular motion ...")
    monkeypatch.setattr(overview_module, "llm", fake_llm)
    guide = describe_notes("s1", "mr")
    assert guide.title == "Class 12 Physics textbook"
    assert "Marathi" in sent[0][0]["content"]
    assert "physics.pdf" in sent[0][1]["content"]


def test_describe_notes_is_empty_when_there_are_no_notes_or_the_model_is_down(monkeypatch):
    monkeypatch.setattr(overview_module, "_sample", lambda session_id: "")
    assert describe_notes("s1", "mr") == Overview()

    def down(*_, **__):
        raise ServiceUnavailable("down")

    monkeypatch.setattr(overview_module, "_sample", lambda session_id: "Files: a.pdf\n\ntext")
    monkeypatch.setattr(overview_module, "llm", down)
    assert describe_notes("s1", "mr") == Overview()
