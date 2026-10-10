"""One turn: VAD -> STT -> agent loop -> language guard -> log, then the voice.

The agent loop is the only part with judgement; every other stage is a single call,
which keeps failures easy to trace in the turn log.

Two entry points: answer_turn() produces and logs the written reply, speak() speaks it. They are separate so the reply can be shown while the voice is still being made.
"""
import json
import logging
import re
import time
from collections.abc import Callable
from types import SimpleNamespace

from agent import stt, tts
from agent.audio import preprocess
from agent.errors import BadClip, ServiceUnavailable
from agent.lang import MATH, detect_language, devanagari_ratio, language_matches
from agent.llm import llm
from agent.prompts import (
    LENGTH_DEFAULT,
    LENGTH_EXACT,
    LENGTH_RETRY,
    OUTLINE_INSTRUCTION,
    PAGE_INSTRUCTION,
    QUIZ_INSTRUCTION,
    REWRITE_IN_LANGUAGE,
    SUMMARY_INSTRUCTION,
    TOOLS,
    TRANSLATE_QUERY,
    phrases,
    system_prompt,
)
from agent.session import Session
from agent.tools.notes import TOP_K, WIDE_K, notes_outline, search_notes
from agent.tools.page import open_page
from agent.tools.quiz import NO_NOTES, quiz_passages
from agent.tools.summary import summary_digest
from agent.turn import Turn, timed
from config import LLM_FALLBACK_MODEL
from db.log import log_speech, log_turn

log = logging.getLogger(__name__)

SUPPORTED_STT_LANGUAGES = frozenset({"marathi", "hindi", "english", "mr", "hi", "en"})
MAX_TOOL_ROUNDS = 2     # a second search is allowed when the first, on a misheard word, finds nothing useful
# A page reference at the end of a reply: "पान 12.", "पेज 12", "(page 12)".
MAX_SENTENCES = 10      # the longest reply a student can ask for
PAGE_REPLY_TOKENS = 1500    # a worked example explained step by step, with its formulas, needs more room than a short answer
SUMMARY_SENTENCES = 8   # every summary is this long: asked to choose, the model picked 3 and left sections out
LONG_REPLY = 4          # a reply of this many sentences or more is given more passages to draw on
BROAD_KINDS = frozenset({"uses", "comparison"})    # questions two sentences cannot answer well
_SENTENCE_BREAK = re.compile(r"(?<=[।.?!])\s+|\n+")
# "example 1.6", "Fig. 2.3", "table 4.1", "उदाहरण 1.6", and "page 12", "पान 12", "page number 12".
_NAMED_ITEM = re.compile(r"(\bexample|\bfigure|\bfig|\btable|उदाहरण)\.?\s*(?:no\.?|number)?\s*(\d{1,2}(?:\.\d{1,2}){1,2})", re.IGNORECASE)
_NAMED_PAGE = re.compile(r"(?:\bpage|\bpg\.?|पान|पेज|पृष्ठ)\s*(?:number|no\.?|क्रमांक|नंबर)?\s*(\d{1,4})\b", re.IGNORECASE)
_AS_PRINTED = {"example": "Example", "उदाहरण": "Example", "figure": "Fig.", "fig": "Fig.", "table": "Table"}


def named_reference(text: str) -> dict | None:
    """The page and the numbered item a question names, as arguments for open_page; None if it names neither."""
    item, page = _NAMED_ITEM.search(text), _NAMED_PAGE.search(text)
    if not item and not page:
        return None
    return {
        "page": int(page.group(1)) if page else None,
        "item": f"{_AS_PRINTED[item.group(1).lower()]} {item.group(2)}" if item else None,
    }
_PAGE_AT_END = re.compile(
    r"[\s(]*(?:पान|पेज|पृष्ठ|pages?)\s*[0-9०-९]+(?:\s*(?:[-‐-―]|ते|से|to)\s*[0-9०-९]+)?\s*\)?\.?\s*$", re.IGNORECASE
)
# The same, but only as a sentence of its own: "...धडे आहेत. पान 12." and not "...starts on page 158."
_PAGE_ALONE_AT_END = re.compile(
    r"(?<=[।.?!\n])\s*(?:पान|पेज|पृष्ठ|pages?)\s*[0-9०-९]+(?:\s*(?:[-‐-―]|ते|से|to)\s*[0-9०-९]+)?\s*\.?\s*$", re.IGNORECASE
)


def format_chunks(chunks: list[dict]) -> str:
    """What the LLM sees as a tool result: each passage labelled with its file and page."""
    return "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in chunks)


def _count(number: int, thing: str) -> str:
    return f"{number} {thing}" if number == 1 else f"{number} {thing}s"


def format_outline(rows: list[dict]) -> str:
    """What the LLM sees from notes_outline: each file, then its chapters with title and pages."""
    lines = []
    for source in dict.fromkeys(row["source"] for row in rows):
        parts = [row for row in rows if row["source"] == source]
        chapters = [row for row in parts if row["chapter"] is not None]
        size = _count(sum(row["pages"] for row in parts), "page")
        lines.append(f"{source}: {size}, {_count(len(chapters), 'chapter') if chapters else 'no chapter headings'}")
        for row in chapters:
            title = f": {row['title']}" if row["title"] else ""
            span = f"page {row['first_page']}" if row["pages"] == 1 else f"pages {row['first_page']} to {row['last_page']}"
            lines.append(f"Chapter {row['chapter']}{title}, {span} ({_count(row['pages'], 'page')})")
    return "\n".join(lines)


def _chapter_number(value) -> int | None:
    """The chapter argument of quiz_me as an int, or None if the model sent nothing usable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _queries_for_notes(session: Session, query: str) -> list[str]:
    """The search query as it should be put to the notes: once for each language they are written in.

    A Marathi or Hindi query finds nothing in an English book, so it gets one small call that
    translates it; if that call fails the query is used as it is. Notes in an Indian language
    are searched with the query as written. Notes in both get both.
    """
    languages = session.notes_languages
    if "en" not in languages or devanagari_ratio(query) < 0.5:
        return [query]
    try:
        message, _ = llm(
            [{"role": "system", "content": TRANSLATE_QUERY}, {"role": "user", "content": query}],
            model=LLM_FALLBACK_MODEL,
            max_tokens=200,
        )
    except ServiceUnavailable:
        return [query]
    english = (message.content or "").strip() or query
    return [english] if languages == ["en"] or english == query else [query, english]


def _merged(results: list[list[dict]], limit: int) -> list[dict]:
    """Several searches as one list: the best of each in turn, without repeats."""
    merged, seen = [], set()
    for rank in range(max(map(len, results), default=0)):
        for found in results:
            if rank < len(found) and (key := (found[rank]["source"], found[rank]["page"], found[rank]["text"][:80])) not in seen:
                seen.add(key)
                merged.append(found[rank])
    return merged[:limit]


def requested_sentences(args: dict | None) -> int | None:
    """The reply length to write, as a number of sentences from 1 to MAX_SENTENCES, or None for the default.

    It is the length the student asked for. A broad question (importance, uses, a comparison) gets
    at least LONG_REPLY sentences, unless the student asked for a single line.
    """
    args = args or {}
    try:
        wanted = min(max(int(args.get("sentences")), 1), MAX_SENTENCES)
    except (TypeError, ValueError):
        wanted = None
    if args.get("kind") in BROAD_KINDS and wanted != 1:
        return max(wanted or 0, LONG_REPLY)
    return wanted


def _unit(count: int) -> str:
    return "sentence" if count == 1 else "sentences"


def count_sentences(reply: str) -> int:
    """Sentences in a reply, not counting the page reference at its end or a formula on a line of its own."""
    body = MATH.sub("", _PAGE_AT_END.sub("", reply.strip()))
    return len([piece for piece in _SENTENCE_BREAK.split(body) if piece.strip()])


def says_not_in_notes(reply: str, language: str) -> bool:
    """Did the model answer that the notes do not cover the question?"""
    return phrases(language)["missing"].rstrip(".") in reply


def run_tool(session: Session, name: str, args: dict, question: str) -> tuple[str, list[dict]]:
    """Execute one tool call. Returns (text for the LLM, passages used)."""
    if name == "search_notes":
        queries = _queries_for_notes(session, args.get("query") or question)
        args["query"] = " | ".join(queries)
        wanted = requested_sentences(args)
        k = WIDE_K if wanted and wanted >= LONG_REPLY else TOP_K
        # A second language gets one more passage, not twice as many.
        chunks = _merged([search_notes(session.id, query, k) for query in queries], k + len(queries) - 1)
        if not chunks:
            return NO_NOTES, chunks
        length = LENGTH_EXACT.format(n=wanted, unit=_unit(wanted)) if wanted else LENGTH_DEFAULT
        return format_chunks(chunks) + length, chunks
    if name == "notes_outline":
        outline = notes_outline(session.id)
        return (format_outline(outline) + OUTLINE_INSTRUCTION if outline else NO_NOTES), []
    if name == "summarise_notes":
        digest, chunks, problem = summary_digest(session.id, _chapter_number(args.get("chapter")))
        if problem:
            return problem, []
        return digest + SUMMARY_INSTRUCTION.format(n=SUMMARY_SENTENCES, unit=_unit(SUMMARY_SENTENCES)), chunks
    if name == "quiz_me":
        chunks, problem = quiz_passages(session.id, _chapter_number(args.get("chapter")))
        return (problem or QUIZ_INSTRUCTION + format_chunks(chunks)), chunks
    return f"UNKNOWN_TOOL: {name} does not exist. Answer without it.", []


def agent_turn(session: Session, turn: Turn) -> None:
    """Agent call (answer or tool call) -> run the tool -> agent call (grounded reply, or one more search).

    Fills in turn.reply_text and the tool, source and token fields, and updates the session's memory.
    The tool and sources recorded are those of the last tool call, the one the reply rests on.
    """
    system = system_prompt(turn.language, session.summary, session.quiz, tuple(session.notes_languages))
    messages = [{"role": "system", "content": system}]
    messages += [*session.messages, {"role": "user", "content": turn.transcript}]

    def call(tools: list[dict] | None = None):
        with timed(turn, "llm_ms"):
            # The reply after a page has been read is long: the steps of a solution and their formulas.
            room = {"max_tokens": PAGE_REPLY_TOKENS} if turn.tool_name == "open_page" else {}
            message, usage = llm(messages, tools=tools, **room)
        turn.tokens_in += usage["tokens_in"]
        turn.tokens_out += usage["tokens_out"]
        turn.model = usage["model"]
        return message

    message = call(TOOLS)
    quiz_passages_text = None
    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        if not message.tool_calls:
            break
        tool_call = message.tool_calls[0]
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        turn.tool_name, turn.tool_args = tool_call.function.name, args
        if turn.tool_name == "search_notes" and (named := named_reference(f"{args.get('query', '')} {turn.transcript}")):
            # A page or a numbered example was named. A search finds one passage of it at best, and the
            # smaller model does not think to open the page, so this is decided here, not left to the model.
            turn.tool_name, turn.tool_args = "open_page", {"ask": turn.transcript, **named}
            args = turn.tool_args
        with timed(turn, "tool_ms"):
            written = False
            if turn.tool_name == "open_page":
                # The page is read as a picture and explained in one step; that explanation is the reply.
                page = _chapter_number(args.get("page"))
                content, chunks, written = open_page(
                    session.id, str(page) if page else None, args.get("item") or None, args.get("ask") or turn.transcript, turn.language
                )
                if chunks and not written:
                    content += PAGE_INSTRUCTION
            else:
                content, chunks = run_tool(session, turn.tool_name, args, turn.transcript)
        turn.sources = [
            {"source": c["source"], "page": c["page"], "score": round(c["score"], 3) if "score" in c else None}
            for c in chunks
        ]
        if turn.tool_name == "quiz_me" and chunks:
            quiz_passages_text = format_chunks(chunks)
        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": content})
        # After the last allowed round no tools are offered: it has to answer now.
        if written:
            message = SimpleNamespace(content=content, tool_calls=None)
            break
        message = call(TOOLS if round_number < MAX_TOOL_ROUNDS else None)

    reply = (message.content or "").strip()
    if reply and not language_matches(reply, turn.language):
        # Language guard: one stricter re-ask if the reply drifted to another language.
        turn.language_retry = True
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": REWRITE_IN_LANGUAGE.format(**phrases(turn.language))})
        reply = (call().content or "").strip() or reply
    if not reply:
        raise ServiceUnavailable("the model returned an empty reply")

    # Length guard: if the student asked for a number of sentences and got another, ask once more.
    # "Not in your notes" is exempt: asking for more sentences there would be asking for a made-up answer.
    wanted = None
    if turn.sources and turn.tool_name == "search_notes":
        wanted = requested_sentences(turn.tool_args)
    elif turn.sources and turn.tool_name == "summarise_notes":
        wanted = SUMMARY_SENTENCES
    if wanted and not says_not_in_notes(reply, turn.language) and (found := count_sentences(reply)) != wanted:
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": LENGTH_RETRY.format(found=found, n=wanted, unit=_unit(wanted))})
        reply = (call().content or "").strip() or reply

    if turn.tool_name == "notes_outline":
        # An answer about the book as a whole has no page to cite; the model tends to tack one on anyway.
        reply = _PAGE_ALONE_AT_END.sub("", reply).strip() or reply
    if quiz_passages_text:
        # A quiz question must not say where the answer is; the model tends to add the page anyway.
        reply = _PAGE_AT_END.sub("", reply).strip() or reply
    turn.reply_text = reply
    # A quiz question stays open for exactly one turn: the next message is checked against its passages.
    session.quiz = {"question": reply, "passages": quiz_passages_text} if quiz_passages_text else None
    session.add_turn(turn.transcript, reply)


def _transcribe(session: Session, wav, hint: str | None) -> tuple[str, str]:
    """Speech to text, with one retry when auto-detect lands on a language we do not support.

    Without a hint, Marathi or Hindi can be heard as Urdu and written in Arabic script,
    which nothing downstream can use. The retry forces the session's language, or Hindi.
    """
    transcript, stt_language = stt.transcribe(wav, hint)
    if hint is None and stt_language not in SUPPORTED_STT_LANGUAGES:
        retry_hint = session.language if session.language in ("mr", "hi") else "hi"
        log.info("speech heard as %r; transcribing again as %s", stt_language, retry_hint)
        transcript, stt_language = stt.transcribe(wav, retry_hint)
    return transcript, stt_language


def answer_turn(
    session: Session,
    *,
    audio_path: str | None = None,
    text: str | None = None,
    hint: str | None = None,
    on_heard: Callable[[str], None] | None = None,
) -> Turn:
    """Run one turn up to the written reply, from a recording (audio_path) or typed text, and log it.

    hint is the language the student says they are speaking ('mr' or 'hi'), or None to let
    the speech model detect it. Raises BadClip if the recording is empty, too short or too long;
    nothing is logged in that case. Any other failure becomes a "sorry, try again" reply
    with the error recorded on the turn. on_heard is called with the transcript as soon as a
    recording has been transcribed, so the page can show the question while the answer is made.
    """
    turn = Turn(session_id=session.id, language=session.language or hint or "hi")
    try:
        stt_language = ""
        if audio_path is not None:
            with timed(turn, "vad_ms"):
                wav = preprocess(audio_path)
            with timed(turn, "stt_ms"):
                turn.transcript, stt_language = _transcribe(session, wav, hint)
            if on_heard:
                on_heard(turn.transcript)
        else:
            turn.transcript = (text or "").strip()
        # Ties go to the language the session has been using, or the one the student picked on the page.
        turn.language = detect_language(turn.transcript, stt_language, session.language or hint or "")
        session.language = turn.language
        agent_turn(session, turn)
    except BadClip:
        raise
    except Exception as error:  # the turn boundary: whatever broke, the student still gets a reply
        log.exception("turn failed for session %s", session.id)
        turn.add_error(f"{type(error).__name__}: {error}")
        turn.reply_text = phrases(turn.language)["sorry"]

    turn.total_ms = int((time.perf_counter() - turn.started) * 1000)
    turn.id = session.last_turn_id = log_turn(turn)
    return turn


def speak(turn: Turn, voice: str = tts.DEFAULT_VOICE) -> str | None:
    """Speak the reply with the chosen voice and return the audio file's path; then log the voice timing.

    A TTS failure must not lose the written reply, so it is recorded on the turn, never
    raised; None is returned in that case.
    """
    path = None
    try:
        if turn.error and (recorded := tts.fallback_audio(turn.language)):
            path = recorded
        else:
            with timed(turn, "tts_ms"):
                path = tts.synthesize(turn.reply_text, turn.language, voice)
    except Exception as error:  # includes model-loading errors from a local backend
        log.warning("TTS failed: %s: %s", type(error).__name__, error)
        turn.add_error(f"TTS failed: {type(error).__name__}: {error}")
    turn.total_ms = int((time.perf_counter() - turn.started) * 1000)
    log_speech(turn)
    return path
