"""A short guide to newly added notes: a title, what they cover, and three questions to try.

One small LLM call per upload, on the fallback model so it does not use the main model's
quota. If it fails, the app simply shows the file name and no suggestions.
"""
import logging
from dataclasses import dataclass, field

from agent.errors import ServiceUnavailable
from agent.llm import llm
from agent.prompts import phrases
from config import LLM_FALLBACK_MODEL
from db.conn import run

log = logging.getLogger(__name__)

SAMPLE_PASSAGES = 5
SAMPLE_CHARS = 500
PROMPT = """Below are the file names of study notes a student uploaded and a few extracts from different parts of them. Reply in exactly this form and nothing else:

TITLE: what the notes are, in English, at most 6 words, such as "Class 12 Physics textbook"
SUMMARY: two sentences in English on what the notes cover as a whole
Q: a question the student could ask, in {name}
Q: another one, in {name}
Q: a third, in {name}

The three questions are simple "what is" or "how does" questions of at most 9 words, each about one idea named in the extracts, written the way a student would say them aloud in {name} ({script} script) with technical terms kept. No numbers or sums."""


@dataclass
class Overview:
    title: str = ""
    summary: str = ""
    questions: list[str] = field(default_factory=list)


def _sample(session_id: str) -> str:
    """File names, then a few passages spread across the notes so the guide reflects all of them."""
    names = run("SELECT DISTINCT source FROM note_chunks WHERE session_id = %s", (session_id,))
    rows = run(
        "SELECT text FROM (SELECT text, row_number() OVER (ORDER BY id) AS n, count(*) OVER () AS total"
        " FROM note_chunks WHERE session_id = %s) numbered"
        " WHERE (n - 1) %% greatest(total / %s, 1) = 0 ORDER BY n LIMIT %s",
        (session_id, SAMPLE_PASSAGES, SAMPLE_PASSAGES),
    )
    if not rows:
        return ""
    files = ", ".join(name for (name,) in names)
    return f"Files: {files}\n\n" + "\n\n".join(text[:SAMPLE_CHARS] for (text,) in rows)


def parse(reply: str) -> Overview:
    """Read the TITLE / SUMMARY / Q lines back out. Anything missing is left empty."""
    overview = Overview()
    for line in reply.splitlines():
        label, _, value = line.partition(":")
        value = value.strip().strip('"')
        match label.strip().upper():
            case "TITLE":
                overview.title = value
            case "SUMMARY":
                overview.summary = value
            case "Q" if value and len(overview.questions) < 3:
                overview.questions.append(value)
    return overview


def describe_notes(session_id: str, language: str) -> Overview:
    """Title, summary and suggested questions for a session's notes; empty on any failure."""
    try:
        sample = _sample(session_id)
        if not sample:
            return Overview()
        message, _ = llm(
            [
                {"role": "system", "content": PROMPT.format(**phrases(language))},
                {"role": "user", "content": sample},
            ],
            model=LLM_FALLBACK_MODEL,
            max_tokens=500,
        )
        return parse(message.content or "")
    except ServiceUnavailable:
        log.warning("could not write an overview of the notes")
        return Overview()
