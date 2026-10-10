"""The summarise_notes tool: a digest of one chapter, or of all the notes, short enough for one LLM call.

A chapter cannot be sent whole (28 pages are about 27,000 tokens), so the digest is its
skeleton: each numbered section's heading with the lines that open it. The digest of all the
notes is each chapter's title with the names of its sections. Notes with no numbered
sections, or with no chapters at all, are sampled evenly from start to end instead.
"""
import re

from agent.tools.notes import available_chapters, chapter_chunks, notes_outline, spread_chunks
from agent.tools.quiz import NO_NOTES

MAX_PARTS = 14          # sampled passages in the digest of a chapter that has no numbered sections
OPENING_CHARS = 260     # the most of the text after a section heading that is included
CHAPTER_CHARS = 3600    # the openings of all of a chapter's sections are kept to about this size
SAMPLE_CHARS = 420      # how much of a sampled passage is included
MIN_SECTIONS = 3        # with fewer headings than this, the chapter is sampled instead
BOOK_CHARS = 6000       # the digest of all the notes is kept to about this size


def _thin(items: list, limit: int) -> list:
    """At most `limit` of the items, evenly spaced and in order."""
    return items[:: max(1, -(-len(items) // limit))][:limit]


def _span(row: dict) -> str:
    return f"page {row['first_page']}" if row["pages"] == 1 else f"pages {row['first_page']} to {row['last_page']}"


def sections(chunks: list[dict], chapter: int) -> list[dict]:
    """The numbered sections of a chapter found in its passages, in order, as {heading, page, opening}.

    A heading is recognised in the form "8.2 Application of Gauss' Law:". Sub-sections
    (8.2.1) are left out, and a heading repeated where two passages overlap is taken once.
    """
    heading = re.compile(rf"(?<![\d.]){chapter}\.(\d{{1,2}})\.?\s+([A-Z][^:.?!;]{{2,70}}):\s*")
    found: dict[int, dict] = {}
    for chunk in chunks:
        for match in heading.finditer(chunk["text"]):
            if int(match.group(1)) not in found:
                found[int(match.group(1))] = {
                    "heading": f"{chapter}.{match.group(1)} {match.group(2).strip()}",
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "opening": chunk["text"][match.end() : match.end() + OPENING_CHARS].strip(),
                }
    return [found[number] for number in sorted(found)]


def _sampled(chunks: list[dict]) -> str:
    return "\n\n".join(f"[page {chunk['page']}] {chunk['text'][:SAMPLE_CHARS]}" for chunk in chunks)


def _chapter_digest(session_id: str, chapter: int, chunks: list[dict]) -> tuple[str, list[dict]]:
    row = next((row for row in notes_outline(session_id) if row["chapter"] == chapter), None)
    head = f"Chapter {chapter}" + (f": {row['title']}" if row and row["title"] else "") + (f", {_span(row)}" if row else "")
    parts = sections(chunks, chapter)
    if len(parts) >= MIN_SECTIONS:
        # Every section is named. A chapter with many sections gets less of each one's opening, never fewer sections.
        room = max(60, min(OPENING_CHARS, CHAPTER_CHARS // len(parts)))
        body = "\n\n".join(f"{part['heading']} (page {part['page']}): {part['opening'][:room]}" for part in parts)
        return f"{head}\nIts sections, each with the lines that open it:\n\n{body}", parts
    sample = _thin(chunks, MAX_PARTS)
    return f"{head}\nPassages from its start to its end:\n\n{_sampled(sample)}", sample


def _notes_digest(session_id: str, outline: list[dict]) -> tuple[str, list[dict]]:
    chapters = [row for row in outline if row["chapter"] is not None]
    if not chapters:
        sample = spread_chunks(session_id, MAX_PARTS)
        return "These notes have no chapter headings. Passages from their start to their end:\n\n" + _sampled(sample), sample
    room = BOOK_CHARS // len(chapters)
    lines = []
    for row in chapters:
        chunks = chapter_chunks(session_id, row["chapter"])
        names = [part["heading"].split(" ", 1)[1] for part in sections(chunks, row["chapter"])]
        about = "Sections: " + "; ".join(names) if names else "It opens: " + (chunks[0]["text"] if chunks else "")
        title = f": {row['title']}" if row["title"] else ""
        lines.append(f"Chapter {row['chapter']}{title} ({_span(row)}). {about}"[:room])
    used = [{"source": row["source"], "page": row["first_page"]} for row in chapters]
    return "All the chapters of the notes, in order:\n\n" + "\n".join(lines), used


def summary_digest(session_id: str, chapter: int | None) -> tuple[str, list[dict], str]:
    """Return (digest, passages it was made from, problem). problem is a note for the LLM when there is nothing to summarise."""
    outline = notes_outline(session_id)
    if not outline:
        return "", [], NO_NOTES
    if chapter is not None:
        chunks = chapter_chunks(session_id, chapter)
        if chunks:
            return (*_chapter_digest(session_id, chapter, chunks), "")
        chapters = available_chapters(session_id)
        if chapters:
            listed = ", ".join(str(number) for number in chapters)
            return "", [], f"NO_CHAPTER: chapter {chapter} is not in the notes. Chapters found: {listed}. Tell the student."
        # Notes with no chapters at all: the only thing there is to summarise is the whole of them.
    return (*_notes_digest(session_id, outline), "")
