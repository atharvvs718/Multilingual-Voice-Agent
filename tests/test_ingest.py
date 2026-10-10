import pytest

import rag.ingest as ingest_module
from rag.ingest import (
    CHUNK_CHARS,
    IngestError,
    Page,
    chunk_pages,
    find_chapter,
    find_chapter_title,
    printed_page_offset,
    read_pages,
)
from rag.pdf_text import looks_garbled

SENTENCE = "प्रकाशसंश्लेषण पानांमध्ये होते. "


# --- chapters -----------------------------------------------------------------------

def test_chapter_heading_at_the_start_of_a_line():
    assert find_chapter("धडा ६: वनस्पतींमधील पोषण") == 6
    assert find_chapter("Chapter 12 Electricity") == 12
    assert find_chapter("प्रकाशसंश्लेषण म्हणजे काय") is None


def test_a_mention_of_another_chapter_is_not_a_heading():
    assert find_chapter("as a ray is called ray optics. (In the Chapter 7 you will study this.)", current=1) is None


def test_numbered_sections_mark_the_chapter():
    assert find_chapter("Can you recall?\n2.1 Introduction:\nIn XIth Std. we discussed solids.", current=1) == 2
    assert find_chapter("some text\n3.4 Mean Free Path:\nmore text", current=3) == 3
    assert find_chapter("16.1 Introduction:\nSemiconductors", current=14) == 16      # a missed chapter is skipped over


def test_a_number_in_a_sum_does_not_start_a_chapter():
    assert find_chapter("The answer is\n9.8 Newton per kilogram as before.", current=2) is None     # too far ahead
    assert find_chapter("mass is\n1.5 Kilogram", current=4) is None                                 # behind


def test_two_sections_on_a_page_correct_a_wrong_chapter():
    page = "5.2 Linear Simple Harmonic Motion:\nbody text\n5.3 Differential Equation of S.H.M.:\nmore"
    assert find_chapter(page, current=9) == 5


# --- printed page numbers -------------------------------------------------------------

def test_printed_page_offset_is_found_from_page_number_lines():
    texts = [(i, "Preface text without a number") for i in range(1, 4)]
    texts += [(i, f"{i - 3}\nBody of the page {i}") for i in range(4, 30)]
    assert printed_page_offset(texts) == 3


def test_no_offset_when_pages_do_not_agree():
    assert printed_page_offset([(i, "Just text.\nMore text.") for i in range(1, 20)]) is None
    assert printed_page_offset([(1, "7\nOnly one numbered page")]) is None


def test_pdf_pages_get_printed_labels_and_front_matter_is_dropped(monkeypatch, tmp_path):
    path = tmp_path / "book.pdf"
    path.write_bytes(b"%PDF")
    texts = [(1, "Preface"), (2, "Contents")] + [(i, f"{i - 2}\nBody text of page {i - 2}.") for i in range(3, 20)]
    monkeypatch.setattr(ingest_module, "_pdf_texts", lambda _path: texts)
    pages = read_pages(path)
    assert [(p.index, p.label) for p in pages[:2]] == [(3, "1"), (4, "2")]
    assert pages[0].text.strip() == "Body text of page 1."      # the page number line is not content
    assert len(pages) == 17


def test_pdf_without_printed_numbers_uses_positions(monkeypatch, tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF")
    monkeypatch.setattr(ingest_module, "_pdf_texts", lambda _path: [(1, "First page."), (2, "Second page.")])
    assert [(p.index, p.label) for p in read_pages(path)] == [(1, "1"), (2, "2")]


# --- chapter titles -------------------------------------------------------------------

def test_title_after_a_chapter_heading():
    assert find_chapter_title("धडा ६: वनस्पतींमधील पोषण\nप्रकाशसंश्लेषण म्हणजे काय?", 6) == "वनस्पतींमधील पोषण"
    assert find_chapter_title("Chapter 12 Electricity\nCharges at rest", 12) == "Electricity"


def test_title_on_a_numbered_line_of_its_own():
    page = (
        "Can you recall?\r\n2.1 Introduction:\r\nIn XIth Std. we discussed solids.\r\n"
        "2. What is atmospheric pressure?\r\n2. Its flow is irrotational: its flow is smooth,\r\n"
        "2. Why does a ball floating on water\r\n2. Mechanical Properties of Fluids\r\nmore text"
    )
    assert find_chapter_title(page, 2) == "Mechanical Properties of Fluids"     # not the questions or the list item
    assert find_chapter_title("10. Magnetic Fields due to Electric Current\r\n10.1 Introduction:", 10) == (
        "Magnetic Fields due to Electric Current"
    )


def test_no_title_when_the_page_does_not_state_one():
    assert find_chapter_title("3.4 Mean Free Path:\nmore text\n3. What are gas laws?", 3) is None
    assert find_chapter_title("2. Mechanical Properties of Fluids", 3) is None       # another chapter's heading


# --- chunks ---------------------------------------------------------------------------

def test_chunks_keep_page_and_inherit_chapter():
    pages = [Page(10, "1", "धडा 6: पोषण\n" + SENTENCE * 60), Page(11, "2", SENTENCE * 10)]
    chunks = chunk_pages(pages)
    assert {(c["page"], c["label"]) for c in chunks} == {(10, "1"), (11, "2")}
    assert all(c["chapter"] == 6 for c in chunks)
    assert all(c["title"] == "पोषण" for c in chunks)       # the title found on the chapter's first page is kept
    assert all(len(c["text"]) <= CHUNK_CHARS for c in chunks)
    assert len([c for c in chunks if c["page"] == 10]) >= 2


def test_chunks_do_not_start_mid_word_and_lines_are_joined():
    words = [f"शब्द{i}" for i in range(400)]
    chunks = chunk_pages([Page(1, "1", "\r\n".join(words))])
    assert len(chunks) > 1
    assert all(c["text"].split()[0] in words for c in chunks)
    assert all("\n" not in c["text"] for c in chunks)


# --- reading --------------------------------------------------------------------------

def test_garbled_detection():
    assert looks_garbled("")                                         # scanned page
    assert looks_garbled("¹ÖÆüÖ ¯ÖÏÛúÖ¿ÖÃÖÓ¿»ÖêÂÖÞÖ ´ÆüÞÖ•Öê ÛúÖµÖ " * 5)   # legacy font
    assert not looks_garbled("प्रकाशसंश्लेषण म्हणजे हिरव्या वनस्पती सूर्यप्रकाशाच्या मदतीने अन्न तयार करतात. " * 3)
    assert not looks_garbled("Photosynthesis is the process by which green plants make food. " * 3)


def test_unsupported_file_type_is_refused(tmp_path):
    path = tmp_path / "notes.docx"
    path.write_bytes(b"x")
    with pytest.raises(IngestError):
        read_pages(path)


def test_file_that_is_not_really_a_pdf_gets_a_clear_message(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(IngestError, match="could not be opened"):
        read_pages(path)


def test_text_file_is_one_page(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("धडा 6: पोषण", encoding="utf-8")
    assert read_pages(path) == [Page(1, "1", "धडा 6: पोषण")]


def test_invisible_joiners_are_not_stored():
    chunks = chunk_pages([Page(1, "1", "तत्\u200dत्\u200dवज्ञानाची ओळख करून घेताना मागच्या वर्षी आपण तीन प्रमुख प्रश्न पाहिले.")])
    assert chunks[0]["text"].startswith("तत्त्वज्ञानाची ओळख")


def test_marathi_queries_become_folded_content_words():
    from agent.tools.notes import _fold, indic_words

    assert _fold("तत्\u200dत्\u200dवज्ञान") == _fold("तत्वज्ञान") == "तत्वज्ञान"         # joiners and the doubled consonant
    assert _fold("महत्त्व") == _fold("महत्व")
    assert indic_words("तत्त्वज्ञान म्हणजे काय आहे") == "तत्वज्ञान:*"                       # asking words are dropped
    assert indic_words("ज्ञानमीमांसा आणि नीतिमीमांसा फरक") == "ज्ञानमीमांसा:* | नीतिमीमांसा:* | फरक:*"
    assert indic_words("surface tension म्हणजे काय") == "surface:* | tension:*"
    assert indic_words("काय आहे") == ""
