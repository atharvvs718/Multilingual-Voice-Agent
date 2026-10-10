"""The open_page tool: read a page of the student's book closely, as a picture.

Text taken out of a PDF loses what matters most on a physics page: equations come out as
scrambled symbols, and figures not at all. So a worked example cannot be explained from the
stored passages. This tool renders the page and has a model that can see read it, which
recovers the equations as they are printed. What that model writes is the reply itself: a
second model rewriting a worked solution only adds mistakes. When the picture cannot be used
(the notes are a text file, the file is no longer stored, the model is busy) the page's
stored text is given to the usual model instead.
"""
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from agent.errors import ServiceUnavailable
from agent.llm import see
from agent.prompts import phrases
from config import CHILD_ENV, ROOT, UPLOADS_DIR
from db.conn import run

log = logging.getLogger(__name__)

RENDER_TIMEOUT_S = 30
MAX_ITEM_CHARS = 60
_LABEL = "coalesce(page_label, page::text)"
NOT_FOUND = "NOT_FOUND: that page or item is not in the notes. Tell the student, or try search_notes."
READ = (
    "This is a page of a student's textbook{more}. The student asks: {ask}\n\n"
    "Answer from what is printed on the page only. For a worked example or a derivation, say what is given and "
    "what is asked, then go through the working one step at a time: the formula used, why it applies, and the "
    "numbers that come out. Copy every equation exactly as printed, in LaTeX between double dollar signs. "
    "Answer exactly what is asked: the whole working, only the formulas, or only what the question means. "
    "If the page does not contain what is asked, say so plainly.\n\n"
    "Write for the student in {name}, in {script} script, keeping technical terms and symbols in English. "
    "Plain short lines only: no headings, no bold, no bullet marks, no tables, no numbering. "
    "Each equation on a line of its own between double dollar signs. At most 16 lines."
)


def stored_file(session_id: str, source: str) -> Path:
    """Where an uploaded file is kept so that its pages can be looked at again."""
    return UPLOADS_DIR / session_id / Path(source).name


def _find(session_id: str, page: str | None, item: str | None) -> tuple[str, int, str] | None:
    """(file, position in the file, printed page) of the item named, else of the page named, else None."""
    if item:
        wanted = "%" + item[:MAX_ITEM_CHARS].replace("\\", "").replace("%", "").replace("_", " ") + "%"
        # Where the book mentions the item more than once, the page the student named decides; otherwise the first.
        rows = run(
            f"SELECT source, page, {_LABEL} FROM note_chunks WHERE session_id = %s AND text ILIKE %s"
            f" ORDER BY ({_LABEL} = %s) DESC, id LIMIT 1",
            (session_id, wanted, page or ""),
        )
        if rows:
            return rows[0]
    if page:
        rows = run(
            f"SELECT source, page, {_LABEL} FROM note_chunks WHERE session_id = %s AND {_LABEL} = %s ORDER BY id LIMIT 1",
            (session_id, page),
        )
        if rows:
            return rows[0]
    return None


def _page_text(session_id: str, source: str, position: int) -> tuple[str, bool]:
    """The stored text of a page, and whether the item on it probably runs on to the next page."""
    rows = run(
        "SELECT text FROM note_chunks WHERE session_id = %s AND source = %s AND page = %s ORDER BY id",
        (session_id, source, position),
    )
    return "\n".join(text for (text,) in rows), bool(rows)


def _pictures(pdf: Path, positions: list[int], folder: str) -> list[bytes]:
    """The pages as JPEGs, rendered by rag.pdf_page in its own process."""
    done = subprocess.run(
        [sys.executable, "-m", "rag.pdf_page", str(pdf), folder, *map(str, positions)],
        capture_output=True,
        timeout=RENDER_TIMEOUT_S,
        cwd=ROOT,
        check=False,
        env=CHILD_ENV,
    )
    if done.returncode != 0:
        log.warning("could not render a page: %s", done.stderr.decode("utf-8", "ignore")[-200:])
        raise ServiceUnavailable("could not render the page")
    return [Path(line).read_bytes() for line in done.stdout.decode().split()]


def open_page(session_id: str, page: str | None, item: str | None, ask: str, language: str) -> tuple[str, list[dict], bool]:
    """Return (text, the pages read, final).

    final=True: text is the finished explanation, written from the picture of the page in the
    student's language. final=False: text is for the LLM to work from, either the page's stored
    text or a note that nothing was found. The page and the one after it are read together,
    because an example often starts near the foot of a page.
    """
    found = _find(session_id, page, item)
    if found is None:
        return NOT_FOUND, [], False
    source, position, label = found
    pdf = stored_file(session_id, source)
    passages = [{"source": source, "page": label}]
    text, _ = _page_text(session_id, source, position)
    next_text, has_next = _page_text(session_id, source, position + 1)
    if pdf.suffix.lower() == ".pdf" and pdf.exists():
        try:
            with tempfile.TemporaryDirectory() as folder:
                pictures = _pictures(pdf, [position, position + 1] if has_next else [position], folder)
            request = READ.format(
                more=" and the page after it" if len(pictures) > 1 else "", ask=(item + ": " if item else "") + ask, **phrases(language)
            )
            return f"{see(request, pictures)}\n{phrases(language)['page']} {label}", passages, True
        except (ServiceUnavailable, subprocess.TimeoutExpired, OSError) as error:
            log.warning("could not look at page %s (%s); using its stored text", label, error)
    stored = text + ("\n" + next_text if has_next else "")
    return f"The stored text of page {label} of {source} (its equations may be scrambled):\n\n{stored}", passages, False
