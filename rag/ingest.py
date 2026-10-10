"""Turn an uploaded .pdf / .txt / .md into chunks in pgvector, each knowing its page and chapter.

Run by hand:  python -m rag.ingest path/to/chapter.pdf --session dev
"""
import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from agent.lang import detect_language
from agent.tools.notes import count_chunks, has_source
from config import CHILD_ENV, MAX_CHUNKS_PER_SESSION, MAX_UPLOAD_MB, NOTES_TTL_HOURS, ROOT
from db.conn import run, run_many
from rag.embed import embed_passages
from rag.pdf_text import CANNOT_OPEN

log = logging.getLogger(__name__)

CHUNK_CHARS = 1000      # about 250-300 tokens, so 3 chunks fit the ~900 token budget
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 40    # shorter pieces are page numbers and stray headers
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PDF_TIMEOUT_S = 180     # a 370-page textbook takes about 3 s; OCR of a scanned one takes far longer

# "धडा 6", "पाठ ६", "प्रकरण 6", "अध्याय 6", "Chapter 6" starting a line near the top of a page. Only ever
# searched in the first 300 characters of a page.
_CHAPTER_HEADING = re.compile(
    r"(?m)^\s*(?:धडा|पाठ|प्रकरण|अध्याय|chapter)\s*[:\-.]?\s*([0-9०-९]{1,2})(?![0-9०-९])", re.IGNORECASE
)
# A numbered section heading at the start of a line: "2.1 Introduction", "10.3.2 Magnetic Force".
# This one is searched through a whole page, and a text file is one page of up to 25 MB. At the start of a
# line it skips blanks but never a line break: "^\s*" there made every line of a file of blank lines start
# a search through all the lines after it.
_SECTION_HEADING = re.compile(r"(?m)^[^\S\n]*(\d{1,2})\.\d{1,2}(?:\.\d{1,2})?\.?:?\s+[A-Z][a-z]{2,}")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_TITLE_MAX_WORDS = 10
TITLE_MAX_CHARS = 120
# What a title never contains: the marks of a question, a list or a running sentence.
_NOT_A_TITLE = re.compile(r"[?:;,।]|\.$")
# A page's own number: a line of digits by itself at the very top or the very bottom. Looked for only in the
# first and last EDGE_CHARS of a page, so a long page costs no more than a short one.
_PAGE_NUMBER_TOP = re.compile(r"\A\s*(\d{1,4})[ \t]*\r?(?=\n|\Z)")
_PAGE_NUMBER_BOTTOM = re.compile(r"(?:\A|\n)[ \t]*(\d{1,4})\s*\Z")
EDGE_CHARS = 200


class IngestError(Exception):
    """Upload could not be indexed. The message is safe to show the user."""


@dataclass
class Page:
    index: int      # position in the file, from 1
    label: str      # what to cite: the number printed on the page when the book has them, else the position
    text: str


@dataclass
class IngestResult:
    source: str
    pages: int
    chunks: int
    chapters: list[int]
    language: str   # 'mr', 'hi' or 'en': the language the notes are written in


# --- Reading ---------------------------------------------------------------------

def _pdf_texts(path: Path) -> list[tuple[int, str]]:
    """[(position in the file, text)] for every readable page, read by rag.pdf_text in its own process."""
    try:
        done = subprocess.run(
            [sys.executable, "-m", "rag.pdf_text", str(path)],
            capture_output=True,
            timeout=PDF_TIMEOUT_S,
            cwd=ROOT,
            check=False,
            env=CHILD_ENV,
        )
    except subprocess.TimeoutExpired as error:
        raise IngestError("This PDF took too long to read.") from error
    if done.returncode != 0:
        log.error("reading %s failed: %s", path.name, done.stderr.decode("utf-8", "ignore")[-500:])
        raise IngestError(CANNOT_OPEN)
    result = json.loads(done.stdout.decode("utf-8"))
    if "error" in result:
        raise IngestError(result["error"])
    return [(index, text) for index, text in result["pages"]]


def printed_page_offset(texts: list[tuple[int, str]]) -> int | None:
    """How far the printed page numbers are behind the positions in the file, if the book has them.

    A textbook's "page 32" is rarely the 32nd page of the PDF. Each page's own number is
    looked for as a line by itself at the top or bottom; if most pages agree on the same
    difference, that difference is the offset.
    """
    offsets = Counter()
    for index, text in texts:
        for number in _page_numbers(text):
            offsets[index - number] += 1
    if not offsets:
        return None
    offset, pages = offsets.most_common(1)[0]
    return offset if pages >= max(5, len(texts) // 2) and offset >= 0 else None


def _own_numbers(text: str) -> list[tuple[int, int, int]]:
    """(number, start, end) of a number standing alone on the first line of a page and on its last.

    Only the two ends of the page are looked at. A match that touches the cut is not trusted:
    the line it is on may go on beyond the part that was looked at.
    """
    found = []
    head = text[:EDGE_CHARS]
    top = _PAGE_NUMBER_TOP.match(head)
    if top and (top.end() < len(head) or len(text) <= EDGE_CHARS):
        found.append((int(top.group(1)), 0, top.end()))
    offset = max(0, len(text) - EDGE_CHARS)
    bottom = _PAGE_NUMBER_BOTTOM.search(text[offset:])
    if bottom:
        line_start = offset + bottom.start() + (1 if text[offset + bottom.start()] == "\n" else 0)
        starts_a_line = line_start == 0 or text[line_start - 1] == "\n"
        beyond_the_top = not found or line_start >= found[0][2]     # a page that is only its number has one number
        if starts_a_line and beyond_the_top:
            found.append((int(bottom.group(1)), line_start, len(text)))
    return found


def _page_numbers(text: str) -> list[int]:
    """The numbers standing alone at the very top and the very bottom of a page (none, one or both)."""
    return [number for number, _, _ in _own_numbers(text)]


def _without_page_number(text: str) -> str:
    """The page's text without its own number, which is not content."""
    found = _own_numbers(text)
    if not found:
        return text
    _, start, end = found[0]
    return text[:start] + text[end:]


def _tidy(text: str) -> str:
    """Undo the PDF's line wrapping (one space between words, lines joined) and drop invisible joiner characters."""
    return re.sub(r"\s+", " ", text.replace("\u200d", "").replace("\u200c", "")).strip()


def read_pages(path: Path) -> list[Page]:
    """The readable pages of a file. Text files are treated as one page."""
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise IngestError(f"File is larger than {MAX_UPLOAD_MB} MB.")
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return [Page(1, "1", path.read_text(encoding="utf-8", errors="ignore"))]
    if suffix != ".pdf":
        raise IngestError("Only .pdf, .txt and .md files are supported.")

    texts = _pdf_texts(path)
    offset = printed_page_offset(texts)
    if offset is None:
        return [Page(index, str(index), text) for index, text in texts]
    # Pages before printed page 1 (preface, pledge, syllabus statements) have no number to cite
    # and are not study content, so they are left out.
    return [
        Page(index, str(index - offset), _without_page_number(text))
        for index, text in texts
        if index > offset
    ]


# --- Chapters and chunks ---------------------------------------------------------

def find_chapter(text: str, current: int | None = None) -> int | None:
    """The chapter a page belongs to, or None if the page does not say.

    Two signs are recognised: a heading such as "Chapter 6" or "धडा ६" starting a line near
    the top, and numbered sections such as "6.2 Stefan's Law". One section number is only
    trusted when it continues from the current chapter, because "9.8 Newton" in a sum looks
    the same; two or more sections of the same chapter on one page are trusted outright.
    """
    sections = Counter(int(match.group(1)) for match in _SECTION_HEADING.finditer(text))
    sections.pop(0, None)
    if sections:
        number, count = sections.most_common(1)[0]
        if count >= 2:
            return number
    heading = _CHAPTER_HEADING.search(text[:300])
    if heading:
        number = int(heading.group(1).translate(_DEV_DIGITS))
        if current is None or number >= current:
            return number
    for number in sections:
        if current is None or current <= number <= current + 2:
            return number
    return None


def _is_title(line: str) -> bool:
    """A short heading: not a question or a piece of a sentence and, in English, mostly capitalised words."""
    words = line.split()
    if not words or len(words) > _TITLE_MAX_WORDS or len(line) > TITLE_MAX_CHARS or _NOT_A_TITLE.search(line):
        return False
    longer = [word for word in words if len(word) > 3 and word[0].isascii() and word[0].isalpha()]
    return sum(word[0].isupper() for word in longer) * 2 >= len(longer)


def find_chapter_title(text: str, number: int) -> str | None:
    """The title of chapter `number`, if this page (the chapter's first) states it.

    Two layouts are recognised: "Chapter 6: Nutrition in Plants" or "धडा ६: वनस्पतींमधील पोषण",
    and a line of its own such as "6. Superposition of Waves". A line numbered the same way
    that is a question or a list item ("6. What are electromagnetic waves?") is not a title.
    """
    heading = _CHAPTER_HEADING.search(text[:300])
    if heading and int(heading.group(1).translate(_DEV_DIGITS)) == number:
        rest = text[heading.end():].split("\n", 1)[0].strip(" \t\r:.-–—")
        if _is_title(rest):
            return rest
    for line in re.finditer(rf"(?m)^[ \t]*{number}\.[ \t]+(\S[^\r\n]*)", text):
        if _is_title(candidate := line.group(1).strip()):
            return candidate
    return None


def chunk_pages(pages: list[Page]) -> list[dict]:
    """Split each page into overlapping chunks, breaking on sentence ends where possible.

    Chunks never cross a page, so every chunk has exactly one page to cite. Each chunk also
    carries its chapter's number and, when the chapter's first page states one, its title.
    """
    chunks, chapter, title = [], None, None
    for page in pages:
        found = find_chapter(page.text, chapter)
        if found and found != chapter:
            chapter, title = found, find_chapter_title(page.text, found)
        text = _tidy(page.text)
        start = 0
        while start < len(text):
            end = min(start + CHUNK_CHARS, len(text))
            if end < len(text):
                cut = max(text.rfind(mark, start + CHUNK_CHARS // 2, end) for mark in ("। ", ". ", "? "))
                if cut != -1:
                    end = cut + 1
            piece = text[start:end].strip()
            if len(piece) > MIN_CHUNK_CHARS:
                chunks.append({"page": page.index, "label": page.label, "chapter": chapter, "title": title, "text": piece})
            if end == len(text):
                break
            start = end - OVERLAP_CHARS
            space = text.find(" ", start, end)      # do not start the next chunk mid-word
            if space != -1:
                start = space + 1
    return chunks


# --- Storing ---------------------------------------------------------------------

def ingest(path: str, session_id: str, source: str | None = None) -> IngestResult:
    """Index one file for one session."""
    file = Path(path)
    name = source or file.name
    if has_source(session_id, name):
        raise IngestError(f"{name} is already added.")
    pages = read_pages(file)
    chunks = chunk_pages(pages)
    if not chunks:
        raise IngestError("No readable text found in the file.")
    if count_chunks(session_id) + len(chunks) > MAX_CHUNKS_PER_SESSION:
        raise IngestError("These notes are too large for one session. Upload fewer or shorter files.")
    vectors = embed_passages([c["text"] for c in chunks])
    run_many(
        "INSERT INTO note_chunks (session_id, source, page, page_label, chapter, chapter_title, text, embedding)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        [
            (session_id, name, c["page"], c["label"], c["chapter"], c["title"], c["text"], vector)
            for c, vector in zip(chunks, vectors, strict=True)
        ],
    )
    sample = " ".join(c["text"] for c in chunks[:: max(1, len(chunks) // 20)])
    result = IngestResult(
        source=name,
        pages=len(pages),
        chunks=len(chunks),
        chapters=sorted({c["chapter"] for c in chunks if c["chapter"] is not None}),
        language=detect_language(sample),
    )
    log.info("indexed %s: %d pages, %d chunks, chapters %s, language %s", name, result.pages, result.chunks,
             result.chapters, result.language)  # fmt: skip
    return result


def delete_session(session_id: str) -> None:
    """Remove everything a session uploaded. Called when the session ends."""
    run("DELETE FROM note_chunks WHERE session_id = %s", (session_id,))


def purge_expired(hours: int = NOTES_TTL_HOURS) -> None:
    """Remove uploads older than the retention limit, for sessions that never ended cleanly."""
    run("DELETE FROM note_chunks WHERE created_at < now() - make_interval(hours => %s)", (hours,))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--session", default="dev")
    args = parser.parse_args()
    print(ingest(args.path, args.session))
