"""The digest a summary is written from: sections of a chapter, chapters of a book, or evenly sampled passages."""
import pytest

import agent.tools.summary as summary

CHAPTER_8 = [
    {"source": "physics.pdf", "page": "186", "chapter": 8, "text": "Can you recall? 8.1 Introduction: In XIth Std we studied Gauss' law. See 9.8 Newton: not ours."},
    {"source": "physics.pdf", "page": "187", "chapter": 8, "text": "more text. 8.2 Application of Gauss' Law: The field of a charged shell is found. 8.2.1 Electric Field Intensity: detail."},
    {"source": "physics.pdf", "page": "188", "chapter": 8, "text": "8.2 Application of Gauss' Law: The field of a charged shell is found again in the overlap."},
    {"source": "physics.pdf", "page": "190", "chapter": 8, "text": "work done. 8.3 Electric Potential and Potential Energy: Work is done against the field."},
]
OUTLINE = [
    {"source": "physics.pdf", "chapter": 8, "title": "Electrostatics", "first_page": "186", "last_page": "213", "pages": 28},
    {"source": "physics.pdf", "chapter": 9, "title": None, "first_page": "214", "last_page": "214", "pages": 1},
]


@pytest.fixture
def notes(monkeypatch):
    """A stand-in notes store: one chapter with numbered sections, one without."""
    plain = [{"source": "physics.pdf", "page": "214", "chapter": 9, "text": f"Passage {i} about current." * 3} for i in range(30)]
    monkeypatch.setattr(summary, "notes_outline", lambda sid: OUTLINE)
    monkeypatch.setattr(summary, "chapter_chunks", lambda sid, chapter: {8: CHAPTER_8, 9: plain}.get(chapter, []))
    monkeypatch.setattr(summary, "available_chapters", lambda sid: [8, 9])
    return plain


def test_sections_are_found_once_in_order_without_sub_sections():
    found = summary.sections(CHAPTER_8, 8)
    assert [part["heading"] for part in found] == ["8.1 Introduction", "8.2 Application of Gauss' Law", "8.3 Electric Potential and Potential Energy"]
    assert found[1]["page"] == "187"                                    # where it first appears, not the overlap
    assert found[1]["opening"].startswith("The field of a charged shell")


def test_a_chapter_with_sections_is_digested_by_section(notes):
    digest, used, problem = summary.summary_digest("s", 8)
    assert problem == ""
    assert digest.startswith("Chapter 8: Electrostatics, pages 186 to 213")
    assert "8.3 Electric Potential and Potential Energy (page 190): Work is done against the field." in digest
    assert [part["page"] for part in used] == ["186", "187", "190"]


def test_a_chapter_with_many_sections_names_them_all_with_shorter_openings(notes, monkeypatch):
    many = [{"source": "physics.pdf", "page": str(200 + i), "chapter": 8, "text": f"8.{i} Topic Number {i}: " + "words " * 80} for i in range(1, 31)]
    monkeypatch.setattr(summary, "chapter_chunks", lambda sid, chapter: many)
    digest, used, _ = summary.summary_digest("s", 8)
    assert len(used) == 30
    assert "8.30 Topic Number 30 (page 230)" in digest
    assert len(digest) < summary.CHAPTER_CHARS + 30 * 40 + 200         # the openings shrink; the headings all stay


def test_a_chapter_without_sections_is_sampled_from_start_to_end(notes):
    digest, used, _ = summary.summary_digest("s", 9)
    assert digest.startswith("Chapter 9, page 214\nPassages from its start to its end:")
    assert len(used) <= summary.MAX_PARTS
    assert used[0] is notes[0]
    assert "Passage 27" in digest                                       # reaches the end of the chapter


def test_all_the_notes_are_digested_chapter_by_chapter(notes):
    digest, used, _ = summary.summary_digest("s", None)
    lines = digest.splitlines()
    assert lines[2] == "Chapter 8: Electrostatics (pages 186 to 213). Sections: Introduction; Application of Gauss' Law; Electric Potential and Potential Energy"
    assert lines[3].startswith("Chapter 9 (page 214). It opens: Passage 0 about current.")
    assert used == [{"source": "physics.pdf", "page": "186"}, {"source": "physics.pdf", "page": "214"}]


def test_a_missing_chapter_and_missing_notes_are_reported(notes, monkeypatch):
    assert summary.summary_digest("s", 40)[2].startswith("NO_CHAPTER: chapter 40 is not in the notes. Chapters found: 8, 9.")
    monkeypatch.setattr(summary, "notes_outline", lambda sid: [])
    assert summary.summary_digest("s", None) == ("", [], summary.NO_NOTES)


def test_notes_without_chapters_are_sampled_whatever_chapter_is_asked_for(monkeypatch):
    passages = [{"source": "my notes.md", "page": "1", "chapter": None, "text": f"Line {i}."} for i in range(5)]
    monkeypatch.setattr(summary, "notes_outline", lambda sid: [{"source": "my notes.md", "chapter": None, "title": None, "first_page": "1", "last_page": "1", "pages": 1}])
    monkeypatch.setattr(summary, "chapter_chunks", lambda sid, chapter: [])
    monkeypatch.setattr(summary, "available_chapters", lambda sid: [])
    monkeypatch.setattr(summary, "spread_chunks", lambda sid, limit: passages)
    for chapter in (None, 3):
        digest, used, problem = summary.summary_digest("s", chapter)
        assert problem == ""
        assert digest.startswith("These notes have no chapter headings.")
        assert used == passages
