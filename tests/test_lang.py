import pytest

from agent.lang import detect_language, devanagari_ratio, language_matches


@pytest.mark.parametrize(
    ("text", "whisper", "expected"),
    [
        ("प्रकाशसंश्लेषण म्हणजे काय?", "marathi", "mr"),
        ("Fundamental rights कितने हैं और कौन से?", "hindi", "hi"),
        ("आणि दुसरा कोणता?", "hindi", "mr"),            # marker words beat a wrong Whisper label
        ("धडा 6 वर मला प्रश्न विचार", "", "mr"),
        ("नमस्ते, तुम क्या कर सकते हो?", "", "hi"),
        ("what is photosynthesis", "english", "en"),
        ("Prakashsanshleshan mhanje kay", "", "mr"),     # Marathi typed in Latin letters
        ("photosynthesis kya hai", "", "hi"),            # Hindi typed in Latin letters
    ],
)
def test_detect_language(text, whisper, expected):
    assert detect_language(text, whisper) == expected


@pytest.mark.parametrize(
    ("text", "whisper", "previous", "expected"),
    [
        ("ऑक्सिजन", "hindi", "mr", "mr"),    # a one-word quiz answer keeps the session's language
        ("ऑक्सिजन", "marathi", "", "mr"),    # no history: trust Whisper
        ("ऑक्सिजन", "", "", "hi"),           # nothing to go on: Devanagari defaults to Hindi
        ("oxygen", "english", "mr", "mr"),   # a short English answer from a Marathi speaker is code-mixing
        ("can you explain this chapter to me in detail", "english", "mr", "en"),
        # Spoken English with "Marathi" chosen: the speech model was told Marathi, but the words are English.
        ("Can you explain the whole thermodynamics chapter in detail?", "mr", "mr", "en"),
        ("Surface tension", "mr", "mr", "mr"),      # two English words from a Marathi speaker are still code-mixing
        ("संपूर्ण पुस्तकाचा सारांश सांग", "", "mr", "mr"),
    ],
)
def test_ties_fall_back_to_session_then_whisper(text, whisper, previous, expected):
    assert detect_language(text, whisper, previous) == expected


def test_devanagari_ratio():
    assert devanagari_ratio("प्रकाश") == 1.0
    assert devanagari_ratio("light") == 0.0
    assert devanagari_ratio("123 ?") == 0.0


@pytest.mark.parametrize(
    ("reply", "expected", "ok"),
    [
        ("प्रकाशसंश्लेषण म्हणजे अन्न तयार करण्याची प्रक्रिया आहे. पान 4.", "mr", True),
        ("Photosynthesis is how plants make food. Page 4.", "mr", False),       # drifted to English
        ("प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे भोजन बनाते हैं. पेज 4.", "mr", False),   # drifted to Hindi
        ("प्रकाश संश्लेषण वह प्रक्रिया है जिसमें पौधे भोजन बनाते हैं. पेज 4.", "hi", True),
        ("Chlorophyll सूर्यप्रकाश शोषून घेते आणि glucose तयार होते. पान 2.", "mr", True),   # English terms are fine
        ("Photosynthesis is how plants make food.", "en", True),
    ],
)
def test_language_guard(reply, expected, ok):
    assert language_matches(reply, expected) is ok


def test_a_formula_does_not_make_a_marathi_reply_look_english():
    reply = "सूत्र असे आहे.\n$$X_L = 2 \\pi f L \\quad \\text{and} \\quad X_C = \\frac{1}{2 \\pi f C}$$\nपान 294"
    assert language_matches(reply, "mr")
    assert not language_matches("The formula is $$F = m a$$ page 4", "mr")
