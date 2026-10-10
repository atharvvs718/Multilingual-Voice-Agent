"""The turn log (FR-14) and feedback (FR-15). Logging must never break a student's turn."""
import logging

import psycopg
from psycopg.types.json import Jsonb

from agent.turn import Turn
from db.conn import run

log = logging.getLogger(__name__)

_INSERT = """
INSERT INTO turns (session_id, language, transcript, tool_name, tool_args, reply_text, model,
                   tokens_in, tokens_out, stt_ms, llm_ms, tool_ms, tts_ms, total_ms, error)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""


def purge_old_turns(days: int) -> None:
    """Delete logged questions and replies older than this many days."""
    run("DELETE FROM turns WHERE created_at < now() - make_interval(days => %s)", (days,))


def forget_turns(session_id: str) -> None:
    """Delete every logged question and reply of one visitor."""
    run("DELETE FROM turns WHERE session_id = %s", (session_id,))


def log_turn(turn: Turn) -> int | None:
    """Store one turn and return its row id, or None if the database is unreachable."""
    try:
        rows = run(
            _INSERT,
            (
                turn.session_id, turn.language, turn.transcript, turn.tool_name,
                Jsonb(turn.tool_args) if turn.tool_args is not None else None,
                turn.reply_text, turn.model, turn.tokens_in, turn.tokens_out,
                turn.stt_ms, turn.llm_ms, turn.tool_ms, turn.tts_ms, turn.total_ms, turn.error,
            ),
        )  # fmt: skip
        return rows[0][0]
    except psycopg.Error:
        log.exception("could not write the turn log")
        return None


def set_feedback(turn_id: int, value: int) -> bool:
    """Record thumbs up (1) or down (-1) on a logged turn. Returns False if it could not be saved."""
    if value not in (1, -1):
        raise ValueError("feedback must be 1 or -1")
    try:
        run("UPDATE turns SET feedback = %s WHERE id = %s", (value, turn_id))
        return True
    except psycopg.Error:
        log.exception("could not save feedback")
        return False


def log_speech(turn: Turn) -> None:
    """Add the voice timing to a turn that was logged before its reply was spoken."""
    if turn.id is None:
        return
    try:
        run(
            "UPDATE turns SET tts_ms = %s, total_ms = %s, error = %s WHERE id = %s",
            (turn.tts_ms, turn.total_ms, turn.error, turn.id),
        )
    except psycopg.Error:
        log.exception("could not update the turn log")
