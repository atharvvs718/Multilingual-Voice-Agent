"""FR-3: decide whether text is Marathi, Hindi or English.

Marathi and Hindi share the Devanagari script, so the script alone cannot tell them
apart. We count very common function words that exist in only one of the two.
"""
import re

MARATHI_WORDS = frozenset({
    "आहे", "आहेत", "काय", "आणि", "नाही", "मला", "तुम्ही", "कोणता", "कोणते", "कसे", "कसा", "कसं",
    "म्हणजे", "वर", "मध्ये", "होते", "होता", "करा", "सांगा", "विचार", "धडा", "किती", "चा", "ची", "चे",
    "हे", "ते", "पान", "तुमच्या", "नोट्समध्ये",
})  # fmt: skip
HINDI_WORDS = frozenset({
    "है", "हैं", "क्या", "और", "नहीं", "मुझे", "आप", "तुम", "कौन", "कैसे", "कितने", "कितना",
    "में", "था", "थी", "करो", "बताओ", "पूछो", "से", "की", "के", "पर", "यह", "पेज", "आपके",
})  # fmt: skip
# The same idea for questions typed in Latin letters ("prakashsanshleshan mhanje kay").
MARATHI_ROMAN = frozenset({
    "mhanje", "kay", "aahe", "ahe", "ahet", "ani", "mala", "kasa", "kase", "kasha", "kuthe", "konta", "konte",
    "sanga", "vichar", "dhada", "madhe", "madhye", "tumhi", "kiti", "nahi", "hota", "hote",
})  # fmt: skip
HINDI_ROMAN = frozenset({
    "kya", "hai", "hain", "aur", "mujhe", "kaise", "kahan", "kaun", "kitne", "kitna", "batao", "bataiye",
    "nahin", "mein", "tum", "aap", "tha", "thi", "karo", "poocho", "kyun", "kyon",
})  # fmt: skip
_LATIN_WORD = re.compile(r"[a-z]+")
_DEVANAGARI_WORD = re.compile(r"[\u0900-\u097F]+")
_DEVANAGARI_CHAR = re.compile(r"[\u0900-\u097F]")
# A formula in a reply, in any of the LaTeX wrappings the model uses: $$..$$, \[..\], \(..\), $..$.
# Each form has a longest length, so a reply full of opening brackets that never close cannot make the search slow.
MATH = re.compile(r"\$\$.{1,600}?\$\$|\\\[.{1,600}?\\\]|\\\(.{1,300}?\\\)|\$[^$\n]{1,300}\$", re.DOTALL)
_STT_NAMES = {"marathi": "mr", "mr": "mr", "hindi": "hi", "hi": "hi", "english": "en", "en": "en"}


def devanagari_ratio(text: str) -> float:
    """Share of the letters in text that are Devanagari (0.0 for text with no letters)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(_DEVANAGARI_CHAR.match(c)) for c in letters) / len(letters)


def _marker_counts(text: str) -> tuple[int, int]:
    """(Marathi-only words, Hindi-only words) found in text."""
    words = _DEVANAGARI_WORD.findall(text)
    roman = _LATIN_WORD.findall(text.lower())
    marathi = sum(w in MARATHI_WORDS for w in words) + sum(w in MARATHI_ROMAN for w in roman)
    hindi = sum(w in HINDI_WORDS for w in words) + sum(w in HINDI_ROMAN for w in roman)
    return marathi, hindi


def detect_language(text: str, stt_language: str = "", previous: str = "") -> str:
    """Return 'mr', 'hi' or 'en'. Code-mixed speech maps to its Indian base language.

    Marker words decide between Marathi and Hindi. When they do not (a one-word quiz answer,
    say), text with any Devanagari keeps the language the session was using, then takes the
    speech model's guess. Text with no Devanagari at all is English, whatever language the
    student said they would speak, unless it is a phrase of up to three words from a student
    who has been speaking Marathi or Hindi: that is code-mixing.
    """
    marathi, hindi = _marker_counts(text)
    if marathi != hindi:
        return "mr" if marathi > hindi else "hi"
    if _DEVANAGARI_CHAR.search(text):
        if previous in ("mr", "hi"):
            return previous
        guess = _STT_NAMES.get(stt_language.lower(), "")
        return guess if guess in ("mr", "hi") else "hi"
    if previous in ("mr", "hi") and len(text.split()) <= 3:
        return previous
    return "en"


def language_matches(reply: str, expected: str) -> bool:
    """Language guard (step 8): is the reply in the language the student used?"""
    reply = MATH.sub(" ", reply)       # a formula is Latin letters in any language
    ratio = devanagari_ratio(reply)
    if expected == "en":
        return ratio < 0.5
    if ratio < 0.5:
        return False        # drifted to English
    marathi, hindi = _marker_counts(reply)
    return marathi >= hindi if expected == "mr" else hindi >= marathi
