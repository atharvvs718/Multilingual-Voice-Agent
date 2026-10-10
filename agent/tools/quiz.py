"""The quiz_me tool (FR-6): pick passages for the LLM to write one question from.

The same passages are kept on the session, so the student's spoken answer on the next
turn is checked against exactly what the question was written from.
"""
import random

from agent.tools.notes import available_chapters, chapter_chunks, random_chunks

QUIZ_PASSAGES = 2
NO_NOTES = "NO_NOTES: the student has not uploaded any notes yet."


def quiz_passages(session_id: str, chapter: int | None) -> tuple[list[dict], str]:
    """Return (passages, problem). problem is empty on success, otherwise a note for the LLM to act on."""
    if chapter is not None:
        chunks = chapter_chunks(session_id, chapter)
        if chunks:
            # A random window, so asking for the same chapter again gives a different question.
            start = random.randrange(max(1, len(chunks) - QUIZ_PASSAGES + 1))
            return chunks[start : start + QUIZ_PASSAGES], ""
        chapters = available_chapters(session_id)
        if chapters:
            listed = ", ".join(str(c) for c in chapters)
            return [], f"NO_CHAPTER: chapter {chapter} is not in the notes. Chapters found: {listed}. Tell the student."
    chunks = random_chunks(session_id, QUIZ_PASSAGES)
    return (chunks, "") if chunks else ([], NO_NOTES)
