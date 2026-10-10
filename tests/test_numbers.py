import pytest

from agent.numbers import number_to_words, spoken_form


@pytest.mark.parametrize(
    ("digits", "marathi", "hindi"),
    [
        ("0", "शून्य", "शून्य"),
        ("6", "सहा", "छह"),
        ("12", "बारा", "बारह"),
        ("29", "एकोणतीस", "उनतीस"),
        ("47", "सत्तेचाळीस", "सैंतालीस"),
        ("99", "नव्व्याण्णव", "निन्यानवे"),
        ("100", "शंभर", "एक सौ"),
        ("112", "एकशे बारा", "एक सौ बारह"),
        ("250", "दोनशे पन्नास", "दो सौ पचास"),
        ("1000", "एक हजार", "एक हज़ार"),
        ("1947", "एक हजार नऊशे सत्तेचाळीस", "एक हज़ार नौ सौ सैंतालीस"),
    ],
)
def test_number_to_words(digits, marathi, hindi):
    assert number_to_words(digits, "mr") == marathi
    assert number_to_words(digits, "hi") == hindi


def test_page_citation_is_spelled_out():
    assert spoken_form("उत्तर पानांमध्ये आहे. पान 12.", "mr") == "उत्तर पानांमध्ये आहे. पान बारा."
    assert spoken_form("यह पत्तों में होता है. पेज 4", "hi") == "यह पत्तों में होता है. पेज चार"


def test_devanagari_digits_and_decimals():
    assert spoken_form("धडा ६", "mr") == "धडा सहा"
    assert spoken_form("3.5 ग्रॅम", "mr") == "तीन पूर्णांक पाच ग्रॅम"
    assert spoken_form("3.5 ग्राम", "hi") == "तीन दशमलव पाँच ग्राम"


def test_long_numbers_are_read_digit_by_digit():
    assert spoken_form("123456", "hi") == "एक दो तीन चार पाँच छह"
    assert spoken_form("कोड 007", "mr") == "कोड शून्य शून्य सात"


def test_english_and_plain_text_are_unchanged():
    assert spoken_form("See page 12.", "en") == "See page 12."
    assert spoken_form("नमस्ते", "hi") == "नमस्ते"
