"""The agent loop and the turn boundary, with the LLM, tools, TTS and database replaced by fakes."""
import pytest

import agent.pipeline as pipeline
import agent.session as session_module
from agent.errors import BadClip, ServiceUnavailable
from agent.prompts import LANGUAGE, system_prompt
from agent.session import Session
from tests.conftest import USAGE, ScriptedLLM, fake_message

CHUNK = {"source": "sci.pdf", "page": 4, "chapter": 6, "text": "प्रकाशसंश्लेषण पानांमध्ये होते.", "score": 0.9}
QUIZ_CHUNK = {"source": "sci.pdf", "page": 5, "chapter": 6, "text": "वनस्पती ऑक्सिजन बाहेर सोडतात."}


@pytest.fixture
def wired(monkeypatch):
    """Patch everything outside the pipeline. Returns a function that installs a scripted LLM."""
    logged, spoken = [], []
    monkeypatch.setattr(pipeline, "search_notes", lambda session_id, query, k: [CHUNK])
    monkeypatch.setattr(pipeline, "quiz_passages", lambda session_id, chapter: ([QUIZ_CHUNK], ""))
    monkeypatch.setattr(pipeline, "log_turn", lambda turn: logged.append(turn) or 7)
    monkeypatch.setattr(pipeline, "log_speech", spoken.append)
    monkeypatch.setattr(pipeline.tts, "synthesize", lambda text, lang, voice: f"{voice}.wav")
    monkeypatch.setattr(pipeline.tts, "fallback_audio", lambda lang: None)

    def install(*replies) -> ScriptedLLM:
        scripted = ScriptedLLM(*replies)
        monkeypatch.setattr(pipeline, "llm", scripted)
        return scripted

    install.logged, install.spoken = logged, spoken
    return install


# --- answer_turn ------------------------------------------------------------------

def test_tool_call_then_grounded_reply(wired):
    scripted = wired(
        fake_message(tool="search_notes", arguments='{"query": "प्रकाशसंश्लेषण"}'),
        fake_message("प्रकाशसंश्लेषण पानांमध्ये होते. पान 4."),
    )
    session = Session(id="s1")
    turn = pipeline.answer_turn(session, text="प्रकाशसंश्लेषण म्हणजे काय?")

    assert turn.tool_name == "search_notes"
    assert turn.tool_args == {"query": "प्रकाशसंश्लेषण"}
    assert turn.sources == [{"source": "sci.pdf", "page": 4, "score": 0.9}]
    assert turn.reply_text == "प्रकाशसंश्लेषण पानांमध्ये होते. पान 4."
    assert (turn.tokens_in, turn.tokens_out, turn.model) == (200, 20, USAGE["model"])
    assert turn.id == session.last_turn_id == 7
    assert wired.logged == [turn]                                 # logged before it is spoken
    second_messages, second_tools = scripted.calls[1]
    assert second_tools is not None                               # one more search is allowed if this one missed
    assert "[sci.pdf, page 4]" in second_messages[-1]["content"]  # the LLM sees the page to cite
    assert [m["role"] for m in session.messages] == ["user", "assistant"]   # tool output is not kept in memory


def test_direct_answer_skips_the_tool(wired, monkeypatch):
    wired(fake_message("नमस्ते! मैं आपके नोट्स से जवाब दे सकता हूँ."))
    monkeypatch.setattr(pipeline, "search_notes", lambda *_: pytest.fail("should not search"))
    turn = pipeline.answer_turn(Session(id="s2"), text="नमस्ते, तुम क्या कर सकते हो?")
    assert turn.tool_name is None
    assert turn.language == "hi"
    assert turn.error is None


def test_no_notes_is_reported_to_the_model(wired, monkeypatch):
    scripted = wired(
        fake_message(tool="search_notes", arguments='{"query": "x"}'),
        fake_message("कृपया नोट्स अपलोड करा आणि मला पुन्हा विचारा."),
    )
    monkeypatch.setattr(pipeline, "search_notes", lambda *_: [])
    pipeline.answer_turn(Session(id="s3"), text="प्रकाशसंश्लेषण म्हणजे काय?")
    assert scripted.calls[1][0][-1]["content"].startswith("NO_NOTES")


def test_quiz_question_stays_open_for_one_turn(wired):
    scripted = wired(
        fake_message(tool="quiz_me", arguments='{"chapter": 6}'),
        fake_message("वनस्पती कोणता वायू बाहेर सोडतात?"),
        fake_message("बरोबर! वनस्पती ऑक्सिजन बाहेर सोडतात. पान 5. आणखी एक प्रश्न हवा आहे का?"),
    )
    session = Session(id="s4")

    asked = pipeline.answer_turn(session, text="धडा 6 वर मला प्रश्न विचार")
    assert asked.tool_name == "quiz_me"
    assert asked.tool_args == {"chapter": 6}
    assert session.quiz["question"] == "वनस्पती कोणता वायू बाहेर सोडतात?"
    assert "ONE short quiz question" in scripted.calls[1][0][-1]["content"]

    checked = pipeline.answer_turn(session, text="ऑक्सिजन")
    assert checked.language == "mr"                              # a one-word answer keeps the session language
    system = scripted.calls[2][0][0]["content"]
    assert "Quiz in progress" in system
    assert "वनस्पती ऑक्सिजन बाहेर सोडतात." in system              # checked against the passages it was written from
    assert session.quiz is None


def test_unusable_chapter_argument_quizzes_the_whole_notes(wired, monkeypatch):
    wired(fake_message(tool="quiz_me", arguments='{"chapter": "six"}'), fake_message("वनस्पती कोणता वायू सोडतात?"))
    asked_for = []
    monkeypatch.setattr(pipeline, "quiz_passages", lambda sid, chapter: (asked_for.append(chapter) or [QUIZ_CHUNK], ""))
    turn = pipeline.answer_turn(Session(id="s5"), text="मला प्रश्न विचार")
    assert asked_for == [None]
    assert turn.error is None


def test_language_guard_re_asks_once(wired):
    scripted = wired(
        fake_message("Photosynthesis happens in the leaves. Page 4."),
        fake_message("प्रकाशसंश्लेषण पानांमध्ये होते. पान 4."),
    )
    turn = pipeline.answer_turn(Session(id="s6"), text="प्रकाशसंश्लेषण म्हणजे काय?")
    assert turn.language_retry is True
    assert turn.reply_text == "प्रकाशसंश्लेषण पानांमध्ये होते. पान 4."
    assert "not in Marathi" in scripted.calls[1][0][-1]["content"]
    assert turn.tokens_in == 200                                   # the extra call is counted


def test_llm_outage_gives_a_sorry_reply_and_is_logged(wired, monkeypatch):
    def down(*_, **__):
        raise ServiceUnavailable("LLM unavailable")

    wired()
    monkeypatch.setattr(pipeline, "llm", down)
    session = Session(id="s7")
    turn = pipeline.answer_turn(session, text="प्रकाशसंश्लेषण म्हणजे काय?")
    assert turn.reply_text == LANGUAGE["mr"]["sorry"]
    assert "ServiceUnavailable" in turn.error
    assert wired.logged == [turn]                                  # failures are logged too
    assert session.messages == []                                  # a failed turn is not remembered


def test_bad_clip_propagates_and_is_not_logged(wired, monkeypatch):
    wired()

    def reject(path):
        raise BadClip("No speech heard. Please try again.")

    monkeypatch.setattr(pipeline, "preprocess", reject)
    with pytest.raises(BadClip):
        pipeline.answer_turn(Session(id="s8"), audio_path="clip.wav")
    assert wired.logged == []


def test_chosen_language_breaks_ties_on_the_first_turn(wired):
    wired(fake_message("प्रकाशसंश्लेषण पानांमध्ये होते."))
    turn = pipeline.answer_turn(Session(id="s9"), text="प्रकाशसंश्लेषण तपशीलवार समजावून सांग", hint="mr")
    assert turn.language == "mr"


def test_speech_heard_as_an_unsupported_language_is_transcribed_again(wired, monkeypatch):
    wired(fake_message("नमस्ते! मैं आपके नोट्स से जवाब दे सकता हूँ."))
    hints = []

    def transcribe(wav, hint):
        hints.append(hint)
        return ("پراکاش", "urdu") if hint is None else ("प्रकाशसंश्लेषण क्या है?", "hindi")

    monkeypatch.setattr(pipeline, "preprocess", lambda path: "wav")
    monkeypatch.setattr(pipeline.stt, "transcribe", transcribe)
    turn = pipeline.answer_turn(Session(id="s10"), audio_path="clip.wav")
    assert hints == [None, "hi"]
    assert turn.transcript == "प्रकाशसंश्लेषण क्या है?"
    assert turn.language == "hi"


def test_explicit_hint_is_never_overridden(wired, monkeypatch):
    wired(fake_message("नमस्ते! मैं आपके नोट्स से जवाब दे सकता हूँ."))
    hints = []

    def transcribe(wav, hint):
        hints.append(hint)
        return "नमस्ते, तुम क्या कर सकते हो?", "urdu"

    monkeypatch.setattr(pipeline, "preprocess", lambda path: "wav")
    monkeypatch.setattr(pipeline.stt, "transcribe", transcribe)
    pipeline.answer_turn(Session(id="s11"), audio_path="clip.wav", hint="hi")
    assert hints == ["hi"]


# --- speak ------------------------------------------------------------------------

def test_reply_is_spoken_and_the_log_row_is_updated(wired):
    wired(fake_message("नमस्ते! मैं आपके नोट्स से जवाब दे सकता हूँ."))
    turn = pipeline.answer_turn(Session(id="s12"), text="नमस्ते, तुम क्या कर सकते हो?")
    assert pipeline.speak(turn, "edge-male") == "edge-male.wav"      # the voice picked in the app is the one used
    assert wired.spoken == [turn]


def test_recorded_sorry_is_played_after_a_failed_turn(wired, monkeypatch):
    wired()
    monkeypatch.setattr(pipeline, "llm", lambda *_, **__: (_ for _ in ()).throw(ServiceUnavailable("down")))
    monkeypatch.setattr(pipeline.tts, "fallback_audio", lambda lang: f"sorry_{lang}.wav")
    monkeypatch.setattr(pipeline.tts, "synthesize", lambda *_: pytest.fail("TTS may be down too"))
    turn = pipeline.answer_turn(Session(id="s13"), text="प्रकाशसंश्लेषण म्हणजे काय?")
    assert pipeline.speak(turn) == "sorry_mr.wav"


def test_tts_failure_is_recorded_not_raised(wired, monkeypatch):
    wired(fake_message("नमस्ते! मैं आपके नोट्स से जवाब दे सकता हूँ."))

    def broken(text, lang, voice):
        raise RuntimeError("no GPU")

    monkeypatch.setattr(pipeline.tts, "synthesize", broken)
    turn = pipeline.answer_turn(Session(id="s14"), text="नमस्ते, तुम क्या कर सकते हो?")
    assert pipeline.speak(turn) is None
    assert turn.reply_text.startswith("नमस्ते")
    assert "TTS failed" in turn.error
    assert wired.spoken == [turn]                                  # the error reaches the log row


# --- memory -----------------------------------------------------------------------

def test_old_messages_are_summarised(monkeypatch):
    monkeypatch.setattr(session_module, "llm", lambda *_, **__: (fake_message("Talked about chapter 6."), USAGE))
    session = Session(id="s15")
    for i in range(5):
        session.add_turn(f"q{i}", f"a{i}")
    assert len(session.messages) == 8
    assert session.messages[0]["content"] == "q1"
    assert session.summary == "Talked about chapter 6."
    assert "Talked about chapter 6." in system_prompt("mr", session.summary)


def test_summary_failure_does_not_break_the_turn(monkeypatch):
    def down(*_, **__):
        raise ServiceUnavailable("down")

    monkeypatch.setattr(session_module, "llm", down)
    session = Session(id="s16")
    for i in range(5):
        session.add_turn(f"q{i}", f"a{i}")
    assert len(session.messages) == 8
    assert session.summary == ""


def test_new_chat_keeps_the_notes_and_language():
    session = Session(id="s17", language="mr", summary="old", quiz={"question": "q", "passages": "p"})
    session.add_turn("q", "a")
    session.message_turns[1] = 5
    session.reset_chat()
    assert (session.messages, session.summary, session.quiz, session.message_turns) == ([], "", None, {})
    assert (session.id, session.language) == ("s17", "mr")


def test_search_queries_are_asked_for_in_the_language_of_the_notes(wired):
    scripted = wired(fake_message("नमस्ते! मी तुमच्या नोट्समधून उत्तर देतो."))
    pipeline.answer_turn(Session(id="s18", notes_languages=["en"]), text="नमस्ते, तुम्ही काय करू शकता?", hint="mr")
    system = scripted.calls[0][0][0]["content"]
    assert "notes are written in English" in system
    assert "notes are written" not in system_prompt("mr")       # nothing is said when no notes are loaded


def test_a_second_search_is_allowed_but_not_a_third(wired, monkeypatch):
    scripted = wired(
        fake_message(tool="search_notes", arguments='{"query": "lens rule"}'),
        fake_message(tool="search_notes", arguments='{"query": "Lenz law"}'),
        fake_message("लेन्झचा नियम प्रेरित प्रवाहाची दिशा सांगतो. पान 268."),
    )
    queries = []
    monkeypatch.setattr(pipeline, "search_notes", lambda sid, query, k: queries.append(query) or [{**CHUNK, "page": 268}])
    turn = pipeline.answer_turn(Session(id="s19"), text="लेंसचा नियम काय सांगतो?", hint="mr")
    assert queries == ["lens rule", "Lenz law"]
    assert turn.tool_args == {"query": "Lenz law"}                # the search the reply rests on
    assert turn.sources[0]["page"] == 268
    assert scripted.calls[2][1] is None                           # after two searches it has to answer
    assert turn.tokens_in == 300


def test_a_quiz_question_never_gives_away_the_page(wired):
    wired(fake_message(tool="quiz_me", arguments='{"chapter": 6}'), fake_message("वनस्पती कोणता वायू बाहेर सोडतात? पान 5."))
    session = Session(id="s20")
    turn = pipeline.answer_turn(session, text="धडा 6 वर मला प्रश्न विचार")
    assert turn.reply_text == "वनस्पती कोणता वायू बाहेर सोडतात?"
    assert session.quiz["question"] == "वनस्पती कोणता वायू बाहेर सोडतात?"


def test_a_marathi_search_query_is_translated_for_an_english_book(wired, monkeypatch):
    scripted = wired(
        fake_message(tool="search_notes", arguments='{"query": "पृष्ठताण म्हणजे काय"}'),
        fake_message("surface tension definition"),                # the translation call
        fake_message("पृष्ठताण म्हणजे द्रवाच्या पृष्ठभागावरील ताण. पान 35."),
    )
    queries = []
    monkeypatch.setattr(pipeline, "search_notes", lambda sid, query, k: queries.append(query) or [CHUNK])
    turn = pipeline.answer_turn(Session(id="s21", notes_languages=["en"]), text="पृष्ठताण म्हणजे काय?", hint="mr")
    assert queries == ["surface tension definition"]
    assert turn.tool_args == {"query": "surface tension definition"}      # the details panel shows what was searched
    assert "Translate this search query" in scripted.calls[1][0][0]["content"]


def test_queries_are_left_alone_when_the_notes_are_not_english(wired, monkeypatch):
    wired(fake_message(tool="search_notes", arguments='{"query": "प्रकाशसंश्लेषण"}'), fake_message("उत्तर. पान 4."))
    queries = []
    monkeypatch.setattr(pipeline, "search_notes", lambda sid, query, k: queries.append(query) or [CHUNK])
    pipeline.answer_turn(Session(id="s22", notes_languages=["mr"]), text="प्रकाशसंश्लेषण म्हणजे काय?", hint="mr")
    assert queries == ["प्रकाशसंश्लेषण"]


# --- reply length -----------------------------------------------------------------

def test_count_sentences_ignores_the_page_reference():
    assert pipeline.count_sentences("पहिले वाक्य. दुसरे वाक्य आहे? तिसरे!\nपान 12.") == 3
    assert pipeline.count_sentences("एक ओळ\nदुसरी ओळ\nतिसरी ओळ") == 3
    assert pipeline.count_sentences("यह एक वाक्य है। यह दूसरा है। पेज 4") == 2


def test_a_formula_on_its_own_line_is_not_counted_as_a_sentence():
    reply = "बल वस्तुमान आणि त्वरण यांच्या गुणाकाराइतके असते.\n$$F = m a$$\nयेथे $F$ म्हणजे बल.\nपान 4."
    assert pipeline.count_sentences(reply) == 2


def test_requested_sentences_is_clamped_and_tolerant():
    assert pipeline.requested_sentences({"sentences": 7}) == 7
    assert pipeline.requested_sentences({"sentences": "5"}) == 5
    assert pipeline.requested_sentences({"sentences": 40}) == pipeline.MAX_SENTENCES
    assert pipeline.requested_sentences({"sentences": 0}) == 1
    assert pipeline.requested_sentences({"query": "x"}) is None
    assert pipeline.requested_sentences({"sentences": "many"}) is None


def test_a_broad_question_gets_a_longer_reply_unless_one_line_was_asked_for():
    assert pipeline.requested_sentences({"kind": "fact"}) is None
    assert pipeline.requested_sentences({"kind": "fact", "sentences": 2}) == 2
    assert pipeline.requested_sentences({"kind": "uses"}) == pipeline.LONG_REPLY
    assert pipeline.requested_sentences({"kind": "comparison", "sentences": 2}) == pipeline.LONG_REPLY   # the model's habitual 2
    assert pipeline.requested_sentences({"kind": "uses", "sentences": 7}) == 7
    assert pipeline.requested_sentences({"kind": "uses", "sentences": 1}) == 1                            # "in one line" wins


def test_the_model_is_told_the_exact_length_the_student_asked_for(wired):
    seven = "\n".join(f"वाक्य क्रमांक {i} येथे आहे." for i in range(1, 8)) + "\nपान 4."
    scripted = wired(fake_message(tool="search_notes", arguments='{"query": "प्रकाशसंश्लेषण", "sentences": 7}'), fake_message(seven))
    turn = pipeline.answer_turn(Session(id="s23"), text="प्रकाशसंश्लेषणाबद्दल 7 ओळींमध्ये सांग", hint="mr")
    assert "exactly 7 sentences" in scripted.calls[1][0][-1]["content"]
    assert pipeline.count_sentences(turn.reply_text) == 7
    assert len(scripted.calls) == 2                                # right first time: no re-ask


def test_a_reply_of_the_wrong_length_is_asked_for_again_once(wired):
    three = "एक वाक्य. दुसरे वाक्य. तिसरे वाक्य. पान 4."
    five = "\n".join(f"वाक्य क्रमांक {i} येथे आहे." for i in range(1, 6)) + "\nपान 4."
    scripted = wired(fake_message(tool="search_notes", arguments='{"query": "x", "sentences": 5}'), fake_message(three), fake_message(five))
    turn = pipeline.answer_turn(Session(id="s24"), text="पाच ओळींमध्ये सांग", hint="mr")
    assert "had 3 sentences" in scripted.calls[2][0][-1]["content"]
    assert pipeline.count_sentences(turn.reply_text) == 5


def test_not_in_your_notes_is_never_stretched_to_the_requested_length(wired):
    scripted = wired(fake_message(tool="search_notes", arguments='{"query": "gravitation", "sentences": 5}'), fake_message("हे तुमच्या नोट्समध्ये नाही."))
    turn = pipeline.answer_turn(Session(id="s26"), text="Gravitation म्हणजे काय? पाच वाक्यात सांगा.", hint="mr")
    assert "do not answer the question" in scripted.calls[1][0][-1]["content"]      # the model is told it may decline
    assert turn.reply_text == "हे तुमच्या नोट्समध्ये नाही."
    assert len(scripted.calls) == 2                                                  # and is not asked for five sentences


def test_without_a_requested_length_the_default_is_stated_and_nothing_is_re_asked(wired):
    scripted = wired(fake_message(tool="search_notes", arguments='{"query": "x"}'), fake_message("उत्तर येथे आहे. पान 4."))
    pipeline.answer_turn(Session(id="s25"), text="प्रकाशसंश्लेषण म्हणजे काय?")
    assert "2 or 3 short sentences" in scripted.calls[1][0][-1]["content"]
    assert len(scripted.calls) == 2


# --- questions about the notes themselves, and broad questions ---------------------------

OUTLINE = [
    {"source": "physics.pdf", "chapter": None, "title": None, "first_page": "1", "last_page": "2", "pages": 2},
    {"source": "physics.pdf", "chapter": 1, "title": "Rotational Dynamics", "first_page": "3", "last_page": "25", "pages": 23},
    {"source": "physics.pdf", "chapter": 2, "title": None, "first_page": "26", "last_page": "26", "pages": 1},
    {"source": "my notes.md", "chapter": None, "title": None, "first_page": "1", "last_page": "1", "pages": 1},
]


def test_outline_lists_each_file_with_its_chapters():
    assert pipeline.format_outline(OUTLINE).splitlines() == [
        "physics.pdf: 26 pages, 2 chapters",
        "Chapter 1: Rotational Dynamics, pages 3 to 25 (23 pages)",
        "Chapter 2, page 26 (1 page)",
        "my notes.md: 1 page, no chapter headings",
    ]


def test_a_question_about_the_book_is_answered_from_the_outline(wired, monkeypatch):
    monkeypatch.setattr(pipeline, "notes_outline", lambda session_id: OUTLINE)
    monkeypatch.setattr(pipeline, "search_notes", lambda *_: pytest.fail("should not search the passages"))
    scripted = wired(fake_message(tool="notes_outline"), fake_message("या पुस्तकात दोन धडे आहेत."))
    turn = pipeline.answer_turn(Session(id="s30"), text="या पुस्तकात किती धडे आहेत?", hint="mr")
    seen = scripted.calls[1][0][-1]["content"]
    assert "physics.pdf: 26 pages, 2 chapters" in seen
    assert "name every chapter" in seen
    assert turn.tool_name == "notes_outline"
    assert turn.sources == []                       # nothing to cite: no source line under the reply
    assert turn.reply_text == "या पुस्तकात दोन धडे आहेत."


def test_the_outline_tool_without_notes_asks_for_an_upload(wired, monkeypatch):
    monkeypatch.setattr(pipeline, "notes_outline", lambda session_id: [])
    scripted = wired(fake_message(tool="notes_outline"), fake_message("कृपया तुमच्या नोट्स अपलोड करा."))
    pipeline.answer_turn(Session(id="s31"), text="किती धडे आहेत?", hint="mr")
    assert scripted.calls[1][0][-1]["content"].startswith("NO_NOTES")


@pytest.mark.parametrize(("arguments", "expected"), [
    ('{"query": "x"}', pipeline.TOP_K),
    ('{"query": "x", "sentences": 2}', pipeline.TOP_K),
    ('{"query": "x", "sentences": 4}', pipeline.WIDE_K),
    ('{"query": "x", "kind": "uses"}', pipeline.WIDE_K),
    ('{"query": "x", "kind": "fact", "sentences": 2}', pipeline.TOP_K),
    ('{"query": "x", "sentences": 7}', pipeline.WIDE_K),
])  # fmt: skip
def test_a_longer_reply_is_given_more_passages(wired, monkeypatch, arguments, expected):
    asked = []
    monkeypatch.setattr(pipeline, "search_notes", lambda sid, query, k: asked.append(k) or [CHUNK])
    wired(fake_message(tool="search_notes", arguments=arguments), fake_message("हे तुमच्या नोट्समध्ये नाही."))
    pipeline.answer_turn(Session(id="s32"), text="capacitor चं महत्त्व काय आहे?", hint="mr")
    assert asked == [expected]


@pytest.mark.parametrize(("reply", "expected"), [
    ("या पुस्तकात एकूण १६ धडे आहेत. पान 12.", "या पुस्तकात एकूण १६ धडे आहेत."),          # a page tacked on: removed
    ("There are sixteen chapters.\npage 54", "There are sixteen chapters."),
    ("या पुस्तकात एकूण १६ धडे आहेत. पान 1‑364.", "या पुस्तकात एकूण १६ धडे आहेत."),
    ("Chapter 7 starts on page 158.", "Chapter 7 starts on page 158."),                  # a page that is the answer: kept
    ("सातवा धडा पान 158 ते 185 आहे.", "सातवा धडा पान 158 ते 185 आहे."),
])  # fmt: skip
def test_an_outline_reply_does_not_end_with_a_made_up_page(wired, monkeypatch, reply, expected):
    monkeypatch.setattr(pipeline, "notes_outline", lambda session_id: OUTLINE)
    wired(fake_message(tool="notes_outline"), fake_message(reply))
    language = "en" if reply[0].isascii() else "mr"
    question = "How many chapters are there?" if language == "en" else "या पुस्तकात किती धडे आहेत?"
    assert pipeline.answer_turn(Session(id="s33"), text=question, hint="mr").reply_text == expected


# --- summaries --------------------------------------------------------------------------

def test_a_summary_is_written_from_the_digest_at_the_summary_length(wired, monkeypatch):
    asked = []
    digest = "Chapter 8: Electrostatics, pages 186 to 213\n8.1 Introduction (page 186): Gauss' law"
    monkeypatch.setattr(pipeline, "summary_digest", lambda sid, chapter: asked.append(chapter) or (digest, [{"source": "physics.pdf", "page": "186"}], ""))
    eight = "\n".join(f"वाक्य क्रमांक {i} येथे आहे." for i in range(1, 9)) + "\nपान 186 ते 213."
    scripted = wired(fake_message(tool="summarise_notes", arguments='{"chapter": 8}'), fake_message(eight))
    turn = pipeline.answer_turn(Session(id="s40"), text="धडा 8 चा सारांश सांग", hint="mr")
    seen = scripted.calls[1][0][-1]["content"]
    assert asked == [8]
    assert digest in seen
    assert "exactly 8 sentences" in seen
    assert pipeline.count_sentences(turn.reply_text) == 8          # the page range at the end is not a sentence
    assert len(scripted.calls) == 2                                # the right length first time: no re-ask
    assert turn.sources == [{"source": "physics.pdf", "page": "186", "score": None}]


def test_a_summary_that_is_too_short_is_asked_for_again(wired, monkeypatch):
    monkeypatch.setattr(pipeline, "summary_digest", lambda sid, chapter: ("digest", [{"source": "n.md", "page": "1"}], ""))
    eight = "\n".join(f"वाक्य क्रमांक {i} येथे आहे." for i in range(1, 9)) + "\nपान 1."
    scripted = wired(fake_message(tool="summarise_notes", arguments='{"chapter": null}'), fake_message("एक वाक्य. पान 1."), fake_message(eight))
    turn = pipeline.answer_turn(Session(id="s41"), text="या नोट्सचा सारांश सांग", hint="mr")
    assert "had 1 sentences" in scripted.calls[2][0][-1]["content"]
    assert pipeline.count_sentences(turn.reply_text) == 8


def test_a_chapter_that_is_not_there_is_reported_not_summarised(wired, monkeypatch):
    monkeypatch.setattr(pipeline, "summary_digest", lambda sid, chapter: ("", [], "NO_CHAPTER: chapter 40 is not in the notes."))
    scripted = wired(fake_message(tool="summarise_notes", arguments='{"chapter": 40}'), fake_message("धडा 40 तुमच्या नोट्समध्ये नाही."))
    pipeline.answer_turn(Session(id="s42"), text="धडा 40 चा सारांश सांग", hint="mr")
    assert scripted.calls[1][0][-1]["content"].startswith("NO_CHAPTER")
    assert len(scripted.calls) == 2


# --- notes in Marathi or Hindi, and notes in two languages -----------------------------------

def test_a_marathi_question_is_put_to_marathi_notes_as_written_and_to_english_notes_in_english(wired, monkeypatch):
    searched = []

    def search(sid, query, k):
        searched.append(query)
        book = "philosophy.pdf" if "तत्त्वज्ञान" in query else "physics.pdf"
        return [{"source": book, "page": str(n), "chapter": None, "text": f"{book} passage {n}", "score": 0.9} for n in (1, 2, 3)]

    monkeypatch.setattr(pipeline, "search_notes", search)
    scripted = wired(
        fake_message(tool="search_notes", arguments='{"query": "तत्त्वज्ञान शाखा", "kind": "fact"}'),
        fake_message("philosophy branches"),            # the translation of the query, by the smaller model
        fake_message("तत्त्वज्ञानाच्या तीन शाखा आहेत. पान 1."),
    )
    session = Session(id="s60", notes_languages=["en", "mr"])
    turn = pipeline.answer_turn(session, text="तत्त्वज्ञानाच्या शाखा कोणत्या?", hint="mr")
    assert searched == ["तत्त्वज्ञान शाखा", "philosophy branches"]
    assert turn.tool_args["query"] == "तत्त्वज्ञान शाखा | philosophy branches"
    assert [(s["source"], s["page"]) for s in turn.sources] == [
        ("philosophy.pdf", "1"), ("physics.pdf", "1"), ("philosophy.pdf", "2"), ("physics.pdf", "2"),
    ]                                                   # the best of each in turn, one more passage than for one language
    assert "more than one language: English and Marathi" in scripted.calls[0][0][0]["content"]


def test_marathi_notes_alone_need_no_translation(wired, monkeypatch):
    searched = []
    monkeypatch.setattr(pipeline, "search_notes", lambda sid, query, k: searched.append(query) or [CHUNK])
    scripted = wired(fake_message(tool="search_notes", arguments='{"query": "ज्ञानमीमांसा"}'), fake_message("उत्तर. पान 4."))
    pipeline.answer_turn(Session(id="s61", notes_languages=["mr"]), text="ज्ञानमीमांसा म्हणजे काय?", hint="mr")
    assert searched == ["ज्ञानमीमांसा"]
    assert len(scripted.calls) == 2


def test_merged_results_take_turns_and_drop_repeats():
    first = [{"source": "a", "page": "1", "text": "x"}, {"source": "a", "page": "2", "text": "y"}]
    second = [{"source": "a", "page": "1", "text": "x"}, {"source": "b", "page": "9", "text": "z"}]
    assert [(c["source"], c["page"]) for c in pipeline._merged([first, second], 3)] == [("a", "1"), ("a", "2"), ("b", "9")]
    assert pipeline._merged([[]], 3) == []
