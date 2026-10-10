"""Write numbers out as words before text to speech.

Indic Parler-TTS mispronounces digits ("पान 12" comes out as noise), and the page number is
the part of a reply the student most needs to hear. The on-screen reply keeps its digits.
"""
import re

MARATHI = ["शून्य", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ", "दहा", "अकरा", "बारा", "तेरा", "चौदा", "पंधरा", "सोळा", "सतरा", "अठरा", "एकोणीस", "वीस", "एकवीस", "बावीस", "तेवीस", "चोवीस", "पंचवीस", "सव्वीस", "सत्तावीस", "अठ्ठावीस", "एकोणतीस", "तीस", "एकतीस", "बत्तीस", "तेहतीस", "चौतीस", "पस्तीस", "छत्तीस", "सदतीस", "अडतीस", "एकोणचाळीस", "चाळीस", "एक्केचाळीस", "बेचाळीस", "त्रेचाळीस", "चव्वेचाळीस", "पंचेचाळीस", "सेहेचाळीस", "सत्तेचाळीस", "अठ्ठेचाळीस", "एकोणपन्नास", "पन्नास", "एक्कावन्न", "बावन्न", "त्रेपन्न", "चोपन्न", "पंचावन्न", "छप्पन्न", "सत्तावन्न", "अठ्ठावन्न", "एकोणसाठ", "साठ", "एकसष्ट", "बासष्ट", "त्रेसष्ट", "चौसष्ट", "पासष्ट", "सहासष्ट", "सदुसष्ट", "अडुसष्ट", "एकोणसत्तर", "सत्तर", "एक्काहत्तर", "बाहत्तर", "त्र्याहत्तर", "चौऱ्याहत्तर", "पंचाहत्तर", "शहात्तर", "सत्त्याहत्तर", "अठ्ठ्याहत्तर", "एकोणऐंशी", "ऐंशी", "एक्याऐंशी", "ब्याऐंशी", "त्र्याऐंशी", "चौऱ्याऐंशी", "पंच्याऐंशी", "शहाऐंशी", "सत्त्याऐंशी", "अठ्ठ्याऐंशी", "एकोणनव्वद", "नव्वद", "एक्याण्णव", "ब्याण्णव", "त्र्याण्णव", "चौऱ्याण्णव", "पंच्याण्णव", "शहाण्णव", "सत्त्याण्णव", "अठ्ठ्याण्णव", "नव्व्याण्णव"]
HINDI = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस", "सैंतीस", "अड़तीस", "उनतालीस", "चालीस", "इकतालीस", "बयालीस", "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस", "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरपन", "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ", "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ", "अड़सठ", "उनहत्तर", "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर", "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर", "उन्यासी", "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी", "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे", "इक्यानवे", "बानवे", "तिरानवे", "चौरानवे", "पंचानवे", "छियानवे", "सत्तानवे", "अट्ठानवे", "निन्यानवे"]
assert len(MARATHI) == len(HINDI) == 100

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_POINT = {"mr": "पूर्णांक", "hi": "दशमलव"}
MAX_SPELLED = 9999      # larger numbers are read digit by digit


def _marathi(n: int) -> str:
    if n < 100:
        return MARATHI[n]
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if n == 100:
            return "शंभर"
        return f"{MARATHI[hundreds]}शे" + (f" {MARATHI[rest]}" if rest else "")
    thousands, rest = divmod(n, 1000)
    return f"{MARATHI[thousands]} हजार" + (f" {_marathi(rest)}" if rest else "")


def _hindi(n: int) -> str:
    if n < 100:
        return HINDI[n]
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return f"{HINDI[hundreds]} सौ" + (f" {HINDI[rest]}" if rest else "")
    thousands, rest = divmod(n, 1000)
    return f"{HINDI[thousands]} हज़ार" + (f" {_hindi(rest)}" if rest else "")


_SPELL = {"mr": _marathi, "hi": _hindi}


def number_to_words(digits: str, lang: str) -> str:
    """'12' -> 'बारा' (mr) or 'बारह' (hi). Very large numbers are read one digit at a time."""
    words = MARATHI if lang == "mr" else HINDI
    if int(digits) > MAX_SPELLED or (len(digits) > 1 and digits[0] == "0"):
        return " ".join(words[int(d)] for d in digits)
    return _SPELL[lang](int(digits))


def spoken_form(text: str, lang: str) -> str:
    """Replace every number in text with words in lang. English text is left unchanged."""
    if lang not in _SPELL:
        return text

    def replace(match: re.Match) -> str:
        whole, _, fraction = match.group().partition(".")
        spoken = number_to_words(whole, lang)
        if fraction:
            digits = MARATHI if lang == "mr" else HINDI
            spoken += f" {_POINT[lang]} " + " ".join(digits[int(d)] for d in fraction)
        return spoken

    return _NUMBER.sub(replace, text.translate(_DEVANAGARI_DIGITS))
