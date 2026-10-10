"""The web app: a small API over the pipeline, and the page in web/ that talks to it.

The page is plain HTML, CSS and JavaScript (web/index.html, style.css, app.js), so its look can
be changed without touching this file. Run with start.bat, or:  python server.py
"""
import hashlib
import hmac
import json
import logging
import math
import queue
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import uvicorn
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from agent import budget, tts
from agent.errors import BadClip
from agent.limits import RateLimiter
from agent.overview import Overview, describe_notes
from agent.pipeline import answer_turn, speak
from agent.prompts import LANGUAGE, phrases
from agent.session import Session
from agent.tools.notes import notes_languages, notes_outline
from agent.tools.page import stored_file
from agent.turn import Turn
from config import (
    ACCESS_CODE,
    ASSETS_DIR,
    DAILY_BUDGET_RUPEES,
    HOST,
    MAX_CLIP_S,
    MAX_QUESTION_CHARS,
    MAX_SPOKEN_PER_HOUR,
    MAX_TURNS_PER_HOUR,
    MAX_UPLOAD_MB,
    MAX_UPLOADS_PER_HOUR,
    NOTES_DIR,
    NOTES_TTL_HOURS,
    OUT_DIR,
    PORT,
    PROXY_HOPS,
    ROOT,
    TURN_LOG_DAYS,
    UPLOADS_DIR,
)
from db.init import init as init_database
from db.log import forget_turns, purge_old_turns, set_feedback
from db.notebooks import forget_notebooks, purge_empty_notebooks, remove_notebook, save_notebook, stored_notebooks
from rag.embed import embed_query
from rag.ingest import IngestError, delete_session, ingest, purge_expired

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("server")

WEB = ROOT / "web"
COOKIE = "visitor"
ACCESS_COOKIE = "access"
LANGUAGES = ("mr", "hi", "en")  # what the student can say they are speaking
LANG_NAMES = {"mr": "Marathi", "hi": "Hindi", "en": "English"}
# What the study-tool buttons ask on the student's behalf.
ACTIONS = {
    "mr": {"quiz": "मला एक प्रश्न विचार", "quiz_chapter": "धडा {n} वर मला एक प्रश्न विचार",
           "summary": "संपूर्ण पुस्तकाचा सारांश सांग", "summary_chapter": "धडा {n} चा सारांश सांग"},
    "hi": {"quiz": "मुझसे एक प्रश्न पूछो", "quiz_chapter": "अध्याय {n} पर मुझसे एक प्रश्न पूछो",
           "summary": "पूरी किताब का सारांश बताओ", "summary_chapter": "अध्याय {n} का सारांश बताओ"},
    "en": {"quiz": "Ask me one question", "quiz_chapter": "Ask me one question on chapter {n}",
           "summary": "Summarise the whole book for me", "summary_chapter": "Summarise chapter {n} for me"},
}  # fmt: skip
UPLOAD_NOTICE = (
    "Upload only material you have the right to use. Your notes are kept for this browser only and deleted after "
    f"{NOTES_TTL_HOURS} hours, or at once with \"Remove everything of mine\". Your recordings are not kept. To answer "
    "you, your notes, questions and voice are sent to Sarvam and Groq, and questions and replies are logged for "
    f"{TURN_LOG_DAYS} days. Fonts and the formula library load from Google and jsDelivr. If you are under 18, use this "
    "with a parent or teacher."
)
VOICE_LIMIT = "You have reached this hour's voice limit. The written answer is on the page."
VOICE_SHARE = "You have used your share of today's voice. The written answer is on the page."
NOTES_PROBLEM = "Notes cannot be used right now. Please try again in a little while."
CROSS_SITE = "This request did not come from the app's own page."
VOICE_BUDGET = "Today's voice budget is used up. Answers continue in writing, and the voice is back tomorrow."
VOICE_FAILED = "The voice could not be made just now. The written answer is on the page."
UPLOAD_LIMIT = "You have added several files in the last hour. Please try again a little later."
NEEDS_CODE = "This demo needs its access code."
NEW_NOTEBOOK = "Untitled notebook"      # a notebook's name until its notes give it one, or the student does
NO_NOTEBOOK = "That notebook is no longer here."
NOTEBOOK_PROBLEM = "Notebooks cannot be changed right now. Please try again in a little while."
NOTEBOOK_LIMIT = "You have made several notebooks in the last hour. Please try again a little later."
# What the page may load, and from where: its own files, the two font and formula libraries, nothing else.
CONTENT_POLICY = (
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net/npm/katex@0.16.11/; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net/npm/katex@0.16.11/; "
    "font-src 'self' data: https://fonts.gstatic.com https://cdn.jsdelivr.net/npm/katex@0.16.11/; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
TITLE_CHARS = 44
MAX_CLIP_BYTES = 6_000_000      # a 30 s browser recording is well under 1 MB
# The most a request's body may be, by path; everything else is a few fields of text. Enforced while the body
# is being read (BodyLimit below), because a sender need not say how much is coming.
SMALL_BODY = 64 * 1024
BODY_LIMITS = {"/api/notes": (MAX_UPLOAD_MB + 1) * 1024 * 1024, "/api/ask": MAX_CLIP_BYTES + SMALL_BODY}
PAGE_FILES = {"app.js": "text/javascript", "style.css": "text/css"}     # the only files served besides the page itself
CLIP_SUFFIX = {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/wav": ".wav"}
_VISITOR_ID = re.compile(r"[0-9a-f]{32}")
_AUDIO_NAME = re.compile(r"(reply_[0-9a-f]{8}|sorry_(mr|hi|en))\.wav")

limiter = RateLimiter(MAX_TURNS_PER_HOUR, window_s=3600)
# The voice is the paid part: spoken replies are capped per visitor here, and in rupees per day by agent/budget.py.
voice_limiter = RateLimiter(MAX_SPOKEN_PER_HOUR, window_s=3600)
upload_limiter = RateLimiter(MAX_UPLOADS_PER_HOUR, window_s=3600)
code_limiter = RateLimiter(10, window_s=3600)       # wrong guesses at the access code, per address
describe_limiter = RateLimiter(20, window_s=3600)   # descriptions of newly added notes (one model call each), per address
notebook_limiter = RateLimiter(30, window_s=3600)   # notebooks made or renamed (one row in the database each), per address
# No one visitor may use more than this share of the day's budget on the voice, so one person cannot end it for all.
VISITOR_SHARE = 0.5
visitor_voice = RateLimiter(int(DAILY_BUDGET_RUPEES * 100 * VISITOR_SHARE), window_s=24 * 3600)     # counted in paise
MAX_VISITORS = 2000     # browsers remembered at once; past it the one away longest is forgotten
MAX_NOTEBOOKS = 8       # notebooks one visitor may have
MAX_CHATS = 50          # conversations kept in one notebook; past it the one unused longest goes
MAX_MESSAGES = 200      # lines kept in one conversation
MAX_NAME_CHARS = 100    # a file name is stored with every passage of the file
MAX_TITLE_CHARS = 60    # a notebook's name


# --- Who is asking --------------------------------------------------------------

@dataclass
class Message:
    """One line of a conversation, as the page shows it."""

    role: str                       # "user" or "assistant"
    text: str
    spoken: bool = False            # a question that was said aloud
    turn: Turn | None = None        # the turn behind a reply: what the voice and the feedback act on
    liked: bool | None = None
    key: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


@dataclass
class Chat:
    """One conversation. Its session is the pipeline's memory of it; the session's id is the notebook's."""

    session: Session
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    messages: list[Message] = field(default_factory=list)


@dataclass
class Notebook:
    """One set of sources with the conversations about them. Nothing in it is seen from another notebook.

    Its id is the name its notes, its uploaded files and its logged turns are stored under.
    """

    id: str
    title: str = NEW_NOTEBOOK
    saved: bool = False             # written to the database, which happens once there is something to keep
    chats: dict[str, Chat] = field(default_factory=dict)        # oldest first
    notes_languages: list[str] = field(default_factory=list)    # the same list every chat's session holds
    guides: dict[tuple, Overview] = field(default_factory=dict)  # the notes' title and summary, per (files, language)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)   # one description of the notes at a time

    def chat(self, chat_id: str) -> Chat:
        """The conversation with this id, or a new one."""
        if chat_id not in self.chats:
            chat = Chat(session=Session(id=self.id, notes_languages=self.notes_languages))
            self.chats[chat.id] = chat
            for unused in list(self.chats)[:-MAX_CHATS]:
                del self.chats[unused]
            return chat
        self.chats[chat_id] = self.chats.pop(chat_id)      # now the most recently used
        return self.chats[chat_id]


@dataclass
class Visitor:
    """One browser, and its notebooks. There is always at least one."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    notebooks: dict[str, Notebook] = field(default_factory=dict)    # oldest first
    seen: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)   # one change to the list of notebooks at a time

    @property
    def owner(self) -> str:
        """The name this visitor's notebooks are listed under, and the id of the first one.

        It is a hash of the cookie value, so that neither the database nor the uploads folder
        holds anything a reader of them could use as the cookie.
        """
        return hashlib.sha256(self.id.encode()).hexdigest()[:32]

    def notebook(self, notebook_id: str) -> Notebook:
        """The notebook with this id, or the first one when none is named. Only ever one of this visitor's own."""
        if not self.notebooks:
            self.notebooks[self.owner] = Notebook(id=self.owner)
        if not notebook_id:
            return next(iter(self.notebooks.values()))
        if notebook_id not in self.notebooks:
            raise HTTPException(404, NO_NOTEBOOK)
        return self.notebooks[notebook_id]

    def chats(self) -> Iterator[Chat]:
        """Every conversation, in every notebook."""
        return (chat for notebook in list(self.notebooks.values()) for chat in list(notebook.chats.values()))

    def find_chat(self, chat_id: str) -> Chat | None:
        return next((chat for chat in self.chats() if chat.id == chat_id), None)


visitors: dict[str, Visitor] = {}
_visitors_lock = threading.Lock()


def _open_notebooks(visitor: Visitor) -> None:
    """Bring back what this browser had before a restart: its notebooks are listed in the database, its notes stored there."""
    try:
        for notebook_id, title in stored_notebooks(visitor.owner)[:MAX_NOTEBOOKS]:
            visitor.notebooks[notebook_id] = Notebook(id=notebook_id, title=title, saved=True)
        # Nothing listed: one notebook under the visitor's own name, where notes added before notebooks existed are.
        # It is not written down until something is put in it, so a browser that only looks costs the database nothing.
        for notebook in [visitor.notebook("")] if not visitor.notebooks else visitor.notebooks.values():
            notebook.notes_languages += notes_languages(notebook.id)
    except psycopg.Error:
        log.warning("the database is not reachable; starting this visitor without notes")
        visitor.notebook("")


def _visitor(request: Request) -> Visitor:
    """The visitor this browser is. The id comes from its cookie (see _identify)."""
    visitor_id = request.state.visitor_id
    with _visitors_lock:
        visitor = visitors.get(visitor_id)
        if visitor is None:
            if len(visitors) >= MAX_VISITORS:
                # One with no conversations first: a flood of new browsers must not push out people who are using the app.
                del visitors[min(visitors, key=lambda key: (any(True for _ in visitors[key].chats()), visitors[key].seen))]
            visitor = visitors[visitor_id] = Visitor(id=visitor_id)
            _open_notebooks(visitor)
    visitor.seen = time.time()
    return visitor


def _address(request: Request) -> str:
    """The address a request really came from.

    X-Forwarded-For is written by whoever sends the request, so on its own it cannot be trusted:
    a new made-up value each time would give a new allowance each time. Only the entry added by
    the hosting service's own proxy is used (the last PROXY_HOPS-th one); with no proxy in
    front, the header is ignored and the connection's own address is used.
    """
    forwarded = ",".join(request.headers.getlist("x-forwarded-for"))     # the header may be sent more than once
    hops = [hop.strip() for hop in forwarded.split(",") if hop.strip()]
    if PROXY_HOPS and len(hops) >= PROXY_HOPS:
        return hops[-PROXY_HOPS]
    return request.client.host if request.client else ""


def _caller(request: Request, visitor: Visitor) -> str:
    """Key for the usage caps: the address, so clearing cookies does not reset them."""
    return _address(request) or visitor.id


# --- What the page is sent ------------------------------------------------------

def _cited(turn: Turn) -> list[dict]:
    """The pages a reply was taken from, per file; nothing for a reply that cites no notes."""
    declined = any(words["missing"].rstrip(".") in turn.reply_text for words in LANGUAGE.values())
    if turn.tool_name not in ("search_notes", "open_page") or not turn.sources or declined or turn.error:
        return []
    pages: dict[str, list[str]] = {}
    for passage in turn.sources:
        found = pages.setdefault(passage["source"], [])
        if str(passage["page"]) not in found:
            found.append(str(passage["page"]))
    return [{"source": source, "pages": found} for source, found in pages.items()]


def _shown(message: Message) -> dict:
    turn = message.turn
    return {
        "key": message.key,
        "role": message.role,
        "text": message.text,
        "spoken": message.spoken,
        "liked": message.liked,
        "cited": _cited(turn) if turn else [],
        "language": turn.language if turn else "",
        "can_speak": turn is not None,
        "can_rate": turn is not None and turn.id is not None,
    }


def _details(turn: Turn) -> dict:
    """The "last turn" panel: what was heard, what was looked up, and how long each stage took."""
    stages = {"vad": turn.vad_ms, "stt": turn.stt_ms, "llm": turn.llm_ms, "search": turn.tool_ms,
              "voice": turn.tts_ms, "total": turn.total_ms}  # fmt: skip
    return {
        "heard": turn.transcript,
        "language": LANG_NAMES[turn.language] + (" (reply was re-asked in this language)" if turn.language_retry else ""),
        "tool": f"{turn.tool_name}({turn.tool_args})" if turn.tool_name else "none",
        "passages": ", ".join(
            f"p.{s['page']}" + (f" ({s['score']})" if s.get("score") is not None else "") for s in turn.sources
        ) or "none",
        "timings": " · ".join(f"{name} {value} ms" for name, value in stages.items() if value),
        "tokens": f"{turn.tokens_in} in / {turn.tokens_out} out · {turn.model or 'no model'}",
        # Only what kind of failure it was. The full text can name the database host or quote a provider's
        # answer about the account; that stays in the server's log.
        "error": turn.error.split(":")[0].strip() if turn.error else "",
    }


def _keep(visitor: Visitor, notebook: Notebook) -> None:
    """Write a notebook to the list in the database, so that it is still there after a restart."""
    save_notebook(notebook.id, visitor.owner, notebook.title)
    notebook.saved = True


def _named(title: str) -> str:
    """A notebook's name as it is kept: one line, nothing unprintable, not too long."""
    readable = title.encode("utf-8", "ignore").decode()         # drops half-characters a browser would never send
    printable = "".join(ch if ch >= " " and ch != "\x7f" else " " for ch in readable)     # the database cannot hold a NUL
    return " ".join(printable.split())[:MAX_TITLE_CHARS].strip()


def _listed(visitor: Visitor) -> list[dict]:
    return [{"id": notebook.id, "title": notebook.title} for notebook in visitor.notebooks.values()]


def _notes(visitor: Visitor, notebook: Notebook, language: str, caller: str) -> dict:
    """One notebook's sources, its heading and the questions to try. Empty when nothing is added."""
    try:
        outline = notes_outline(notebook.id)
    except psycopg.Error:
        log.warning("the database is not reachable; showing no notes")
        return {"sources": [], "problem": NOTES_PROBLEM}
    if not outline:
        return {"sources": []}
    sources = []
    for name in dict.fromkeys(row["source"] for row in outline):
        parts = [row for row in outline if row["source"] == name]
        sources.append({
            "source": name,
            "pages": sum(row["pages"] for row in parts),
            "chapters": [row["chapter"] for row in parts if row["chapter"] is not None],
        })  # fmt: skip
    # One description per language for the current files: switching language back and forth costs no further calls.
    made_for = (tuple(s["source"] for s in sources), language)
    with notebook.lock:      # many requests at once still make one description, not one each
        if made_for not in notebook.guides:
            notebook.guides = {key: guide for key, guide in notebook.guides.items() if key[0] == made_for[0]}
            # Past the hourly allowance the notes go undescribed: the file's name stands in for a title.
            notebook.guides[made_for] = describe_notes(notebook.id, language) if describe_limiter.allow(caller) else Overview()
        guide = notebook.guides[made_for]
        # A notebook nobody has named takes its name from what is in it.
        if notebook.title == NEW_NOTEBOOK and (found := _named(guide.title or Path(sources[0]["source"]).stem)):
            notebook.title = found
            try:
                _keep(visitor, notebook)
            except psycopg.Error:
                log.warning("could not save a notebook's name; it keeps the name until the app restarts")
    return {
        "sources": sources,
        "title": notebook.title,        # the page's heading is the notebook's name: one name, and renaming changes both
        "summary": guide.summary,
        "questions": guide.questions[:3],
        "chapters": sorted({chapter for s in sources for chapter in s["chapters"]}),
        "pages": sum(s["pages"] for s in sources),
    }


def _language(value: str) -> str:
    return value if value in LANGUAGES else "mr"


def _title(turn: Turn) -> str:
    """A short name for a conversation, from its first question: the topic searched for, else the words asked."""
    topic = (turn.tool_args or {}).get("query") if turn.tool_name == "search_notes" else ""
    title = " ".join(str(topic or turn.transcript or "New chat").split())
    title = title[:1].upper() + title[1:]
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1].rstrip() + "…"


# --- The app --------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Runs when the server starts, however it was started: the tables, the model warm-up and the hourly clean-up."""
    try:
        init_database()     # creates the tables, and adds columns introduced since the database was made
    except psycopg.Error:
        log.warning("the database is not reachable; notes cannot be added until it is started")
    threading.Thread(target=_warm_up, daemon=True).start()
    threading.Thread(target=_purge_forever, daemon=True).start()
    yield


app = FastAPI(title="Voice Study Companion", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)


# Signs the access pass. It is made new each time the app starts and never leaves the server, so a pass cannot
# be worked out from a guess at the code: the only way to test a guess is /api/access, which counts wrong ones.
_SECRET = secrets.token_bytes(32)


def _access_token(visitor_id: str) -> str:
    """What the access cookie must hold for this browser: a pass only this server can make, for this browser's id."""
    return hmac.new(_SECRET, visitor_id.encode(), hashlib.sha256).hexdigest()


def _may_enter(request: Request) -> bool:
    """With no access code set the app is open; with one, the browser must have given it."""
    given = request.cookies.get(ACCESS_COOKIE, "").encode()
    return not ACCESS_CODE or hmac.compare_digest(given, _access_token(request.state.visitor_id).encode())


def _cookie(request: Request) -> dict:
    """How the app's cookies are set: unreadable by scripts, sent only to this site, and only over HTTPS when the site is."""
    https = request.headers.get("x-forwarded-proto", request.scope.get("scheme", "http")) == "https"
    return {"max_age": NOTES_TTL_HOURS * 3600, "httponly": True, "samesite": "lax", "secure": https}


def _route_path(scope: dict) -> str:
    """The path the router will use. Taken from the request itself, never from request.url: that is rebuilt from
    the Host header, which the sender controls, and a crafted Host made its path come out empty."""
    path, root = scope["path"], scope.get("root_path", "")
    return path[len(root):] if root and path.startswith(root) else path


def _cross_site(request: Request) -> bool:
    """Was this request made by another website's page? Browsers say so; anything else (a script, a test) does not."""
    site = request.headers.get("sec-fetch-site")
    if site:
        return site not in ("same-origin", "none")
    origin = request.headers.get("origin")
    return bool(origin) and urlsplit(origin).netloc != request.headers.get("host", "")


@app.middleware("http")
async def _identify(request: Request, call_next):
    """Give every browser an id in a cookie, ask for the access code if there is one, and set the safety headers."""
    visitor_id = request.cookies.get(COOKIE, "")
    request.state.visitor_id = visitor_id if _VISITOR_ID.fullmatch(visitor_id) else uuid.uuid4().hex
    path = _route_path(request.scope)
    if request.method not in ("GET", "HEAD", "OPTIONS") and _cross_site(request):
        # Another site may not delete notes, add files or ask questions as this visitor, nor be handed a new id for them.
        return JSONResponse({"message": CROSS_SITE}, status_code=403)
    if not path.startswith("/api/") or path == "/api/access" or _may_enter(request):
        response = await call_next(request)
    elif path == "/api/state":
        response = JSONResponse({"locked": True})
    else:
        response = JSONResponse({"message": NEEDS_CODE}, status_code=401)
    if path.startswith("/api/"):
        response.set_cookie(COOKIE, request.state.visitor_id, **_cookie(request))
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers["Cache-Control"] = "no-cache"
    response.headers["Content-Security-Policy"] = CONTENT_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.post("/api/access")
def enter(request: Request, code: str = Body(..., embed=True)):
    """Check the access code; a browser that gives the right one gets a cookie that lets it in."""
    visitor = request.state.visitor_id
    address = _address(request) or visitor
    if not code_limiter.room(address):
        return JSONResponse({"message": "Too many wrong tries. Please wait an hour."}, status_code=429)
    if not ACCESS_CODE or not hmac.compare_digest(code.encode("utf-8", "surrogatepass"), ACCESS_CODE.encode()):
        code_limiter.allow(address)     # only wrong guesses are counted, so a class entering the right code is not locked out
        return JSONResponse({"message": "That is not the access code."}, status_code=403)
    response = JSONResponse({"ok": True})
    response.set_cookie(ACCESS_COOKIE, _access_token(visitor), **_cookie(request))
    return response


@app.get("/")
def page() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/static/{name}")
def page_file(name: str) -> FileResponse:
    """The page's script and stylesheet, by name from a fixed list.

    Not a general "serve this folder": on Windows a path such as \\\\host\\share in the address made the
    folder-serving code try to open a network share.
    """
    if name not in PAGE_FILES:
        raise HTTPException(404)
    return FileResponse(WEB / name, media_type=PAGE_FILES[name])


class _TooLarge(Exception):
    """A request body passed the limit for its path; raised from inside the read, which stops it."""


class BodyLimit:
    """Stop reading a request once its body passes the limit for its path, whether or not it said how long it is."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.inner(scope, receive, send)
            return
        limit = BODY_LIMITS.get(_route_path(scope), SMALL_BODY)
        read = 0
        answering = False

        async def counted():
            nonlocal read
            message = await receive()
            if message["type"] == "http.request":
                read += len(message.get("body", b""))
                if read > limit:
                    raise _TooLarge
            return message

        async def watched(message):
            nonlocal answering
            answering = answering or message["type"] == "http.response.start"
            await send(message)

        # What matters is that the reading stops. Where the framework reads the body itself it reports the
        # interruption as "400, could not parse the body"; where the error reaches here, the answer is 413.
        try:
            await self.inner(scope, counted, watched)
        except* _TooLarge:
            if not answering:
                refusal = JSONResponse({"error": "That is too large.", "message": "That is too large."}, status_code=413)
                await refusal(scope, receive, send)


app.add_middleware(BodyLimit)


@app.get("/api/state")
def state(request: Request, language: str = "mr", notebook: str = "") -> dict:
    """Everything the page needs to show one notebook: its notes, its conversations and the latest one in full,
    with the list of notebooks and the voices. The notebook is the one the page had open, else the first."""
    visitor = _visitor(request)
    current = visitor.notebooks.get(notebook) or visitor.notebook("")
    chats = [chat for chat in current.chats.values() if chat.messages]
    notes = _notes(visitor, current, _language(language), _caller(request, visitor))      # may give the notebook its name
    return {
        "notebooks": _listed(visitor),
        "notebook": current.id,
        "max_notebooks": MAX_NOTEBOOKS,
        "notes": notes,
        "voices": [{"id": speaker, "label": label} for label, speaker in tts.available_voices()],
        "voice": tts.DEFAULT_VOICE,
        "chats": [{"id": chat.id, "title": chat.title} for chat in reversed(chats)],
        "chat": _chat(chats[-1]) if chats else None,
        "notice": UPLOAD_NOTICE,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_clip_s": MAX_CLIP_S,
        "max_question_chars": MAX_QUESTION_CHARS,
    }


def _chat(chat: Chat) -> dict:
    return {"id": chat.id, "title": chat.title, "messages": [_shown(message) for message in chat.messages]}


@app.get("/api/chats/{chat_id}")
def open_chat(request: Request, chat_id: str) -> dict:
    chat = _visitor(request).find_chat(chat_id)
    if chat is None:
        raise HTTPException(404, "That conversation is no longer here.")
    return _chat(chat)


# --- Notebooks ------------------------------------------------------------------

def _spoken_files(notebook: Notebook) -> Iterator[Path]:
    """The audio made for a notebook's replies; never the recorded "sorry" clips."""
    for chat in notebook.chats.values():
        for message in chat.messages:
            for made in (message.turn.spoken.values() if message.turn else ()):
                if Path(made).parent == OUT_DIR:
                    yield Path(made)


def _empty(notebook: Notebook) -> None:
    """Delete what a notebook holds: its notes, its logged questions and replies, the stored files, its spoken replies."""
    delete_session(notebook.id)
    forget_turns(notebook.id)
    shutil.rmtree(UPLOADS_DIR / notebook.id, ignore_errors=True)
    for made in _spoken_files(notebook):
        made.unlink(missing_ok=True)


@app.post("/api/notebooks")
def new_notebook(request: Request, title: str = Body("", embed=True)):
    """Start a notebook: empty, with no sources and no conversations of its own yet."""
    visitor = _visitor(request)
    with visitor.lock:
        if len(visitor.notebooks) >= MAX_NOTEBOOKS:
            return JSONResponse({"message": f"You can have {MAX_NOTEBOOKS} notebooks. Delete one to make room."}, status_code=400)
        if not notebook_limiter.allow(_caller(request, visitor)):
            return JSONResponse({"message": NOTEBOOK_LIMIT}, status_code=429)
        made = Notebook(id=uuid.uuid4().hex, title=_named(title) or NEW_NOTEBOOK)
        try:
            for notebook in [*visitor.notebooks.values(), made]:        # the ones before it too, so their order is kept
                if not notebook.saved:
                    _keep(visitor, notebook)
        except psycopg.Error:
            log.exception("could not save a new notebook")
            return JSONResponse({"message": NOTEBOOK_PROBLEM}, status_code=503)
        visitor.notebooks[made.id] = made
    return {"notebook": made.id, "notebooks": _listed(visitor)}


@app.post("/api/notebooks/{notebook_id}/rename")
def rename_notebook(request: Request, notebook_id: str, title: str = Body(..., embed=True)):
    visitor = _visitor(request)
    notebook = visitor.notebook(notebook_id)
    title = _named(title)
    if not title:
        return JSONResponse({"message": "A notebook needs a name."}, status_code=400)
    if not notebook_limiter.allow(_caller(request, visitor)):
        return JSONResponse({"message": NOTEBOOK_LIMIT}, status_code=429)
    with visitor.lock:
        before, notebook.title = notebook.title, title
        try:
            _keep(visitor, notebook)
        except psycopg.Error:
            notebook.title = before
            log.exception("could not rename a notebook")
            return JSONResponse({"message": NOTEBOOK_PROBLEM}, status_code=503)
    return {"notebook": notebook.id, "notebooks": _listed(visitor)}


@app.post("/api/notebooks/{notebook_id}/delete")
def delete_notebook(request: Request, notebook_id: str):
    """Delete one notebook with everything in it. The other notebooks are not touched."""
    visitor = _visitor(request)
    with visitor.lock:
        notebook = visitor.notebook(notebook_id)
        try:
            _empty(notebook)
            remove_notebook(notebook.id, visitor.owner)
        except psycopg.Error:
            log.exception("could not delete a notebook")
            return JSONResponse({"message": NOTEBOOK_PROBLEM}, status_code=503)
        del visitor.notebooks[notebook.id]
        return {"notebook": visitor.notebook("").id, "notebooks": _listed(visitor)}     # never left without one


# --- Notes ----------------------------------------------------------------------

def _added(visitor: Visitor, notebook: Notebook, language: str) -> None:
    """Notes in this language are now in the notebook, which from here on is worth keeping."""
    if language not in notebook.notes_languages:
        notebook.notes_languages.append(language)
    if not notebook.saved:
        try:
            _keep(visitor, notebook)
        except psycopg.Error:
            log.warning("could not list a notebook; its notes are stored and it is listed at the next change")


@app.post("/api/notes")
def add_notes(request: Request, file: UploadFile = File(...), language: str = Form("mr"), notebook: str = Form("")):  # noqa: B008
    """Index an uploaded .pdf, .txt or .md into one of this visitor's notebooks."""
    visitor = _visitor(request)
    into = visitor.notebook(notebook)
    if not upload_limiter.allow(_caller(request, visitor)):
        return JSONResponse({"error": UPLOAD_LIMIT}, status_code=429)
    name = Path(file.filename or "notes").name
    if len(name) > MAX_NAME_CHARS:
        name = Path(name).stem[: MAX_NAME_CHARS - 10] + Path(name).suffix[:10]
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix.lower(), delete=False) as saved:
        while block := file.file.read(1024 * 1024):
            saved.write(block)
            if saved.tell() > limit:
                break       # ingest refuses it by size; there is no need to store the rest
    try:
        result = ingest(saved.name, into.id, source=name)
    except IngestError as error:
        return JSONResponse({"error": str(error)}, status_code=400)
    except psycopg.Error:
        log.exception("upload failed")
        return JSONResponse({"error": "Could not save your notes just now. Please try again."}, status_code=503)
    else:
        # Kept, so that a page can be looked at as a picture later (agent/tools/page.py); deleted with the notes.
        try:
            kept = stored_file(into.id, name)
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved.name, kept)
        except OSError:
            log.exception("could not keep the uploaded file; its pages cannot be looked at, the notes still work")
    finally:
        Path(saved.name).unlink(missing_ok=True)
    _added(visitor, into, result.language)
    notes = _notes(visitor, into, _language(language), _caller(request, visitor))
    return {"status": f"Added {name}.", "notes": notes, "notebooks": _listed(visitor)}


@app.post("/api/notes/sample")
def add_sample_notes(request: Request, language: str = Form("mr"), notebook: str = Form("")):
    """Load the starter notes that ship with the app, for visitors with nothing to upload."""
    visitor = _visitor(request)
    into = visitor.notebook(notebook)
    if not upload_limiter.allow(_caller(request, visitor)):
        return JSONResponse({"error": UPLOAD_LIMIT}, status_code=429)
    added = 0
    try:
        for path in sorted(NOTES_DIR.glob("*.md")):
            try:
                result = ingest(str(path), into.id)
            except IngestError:
                continue    # already added
            added += 1
            _added(visitor, into, result.language)
    except psycopg.Error:
        log.exception("sample notes failed")
        return JSONResponse({"error": "Could not load the sample notes just now. Please try again."}, status_code=503)
    status = "Added the sample notes." if added else "The sample notes are already added."
    notes = _notes(visitor, into, _language(language), _caller(request, visitor))
    return {"status": status, "notes": notes, "notebooks": _listed(visitor)}


@app.post("/api/forget")
def forget(request: Request):
    """Delete everything this browser has added, now: every notebook with its notes, stored files, conversations,
    logged questions and spoken replies."""
    visitor = _visitor(request)
    with visitor.lock:
        try:
            for notebook in visitor.notebooks.values():
                _empty(notebook)
            forget_notebooks(visitor.owner)
        except psycopg.Error:
            log.exception("could not delete a visitor's notes")
            return JSONResponse({"message": "Could not remove your notes just now. Please try again."}, status_code=503)
    with _visitors_lock:
        visitors.pop(visitor.id, None)
    # A new id from here on, so nothing ties this browser to what it had before.
    had_access = bool(ACCESS_CODE) and _may_enter(request)
    request.state.visitor_id = uuid.uuid4().hex
    response = JSONResponse({"ok": True})
    if had_access:
        response.set_cookie(ACCESS_COOKIE, _access_token(request.state.visitor_id), **_cookie(request))
    return response


# --- A turn ---------------------------------------------------------------------

def _events(work: Callable[[Callable[[dict], None]], None]) -> Iterator[bytes]:
    """Run work(emit) on its own thread and pass on what it emits, one line of JSON each, as it happens."""
    emitted: queue.Queue[dict | None] = queue.Queue()

    def run() -> None:
        try:
            work(emitted.put)
        except Exception:
            log.exception("a turn failed outside the pipeline")
            emitted.put({"event": "error", "message": "Something went wrong. Please try again."})
        finally:
            emitted.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (event := emitted.get()) is not None:
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode()


def _save_clip(audio: UploadFile) -> str:
    """Write the recording the browser sent to a temporary file and return its path."""
    kind = (audio.content_type or "").split(";")[0]
    data = audio.file.read(MAX_CLIP_BYTES + 1)
    if not kind.startswith("audio/") or not data or len(data) > MAX_CLIP_BYTES:
        raise BadClip("That recording could not be read. Please try again.")
    with tempfile.NamedTemporaryFile(suffix=CLIP_SUFFIX.get(kind, ".webm"), delete=False) as file:
        file.write(data)
    return file.name


@app.post("/api/ask")
def ask(
    request: Request,
    chat: str = Form(""),
    notebook: str = Form(""),
    language: str = Form("mr"),
    text: str = Form(""),
    action: str = Form(""),
    chapter: str = Form(""),
    audio: UploadFile | None = File(None),  # noqa: B008
) -> StreamingResponse:
    """One question: typed (text), said aloud (audio) or sent by a study tool (action "quiz" or "summary").

    It is asked of one notebook: only that notebook's notes are searched, and the conversation is one of its own.
    The answer comes back as lines of JSON: {"event": "heard"} with the words as soon as a
    recording is transcribed or a tool's question is chosen, then {"event": "reply"} with the
    written reply, or {"event": "error"} with a message to show.
    """
    visitor = _visitor(request)
    caller = _caller(request, visitor)
    asked_of = visitor.notebook(notebook)
    language = _language(language)
    if action in ("quiz", "summary"):
        wording = ACTIONS[language]
        numbered = chapter.isdigit() and len(chapter) <= 2
        text = wording[f"{action}_chapter"].format(n=chapter) if numbered else wording[action]
    text = text.strip()
    if len(text) > MAX_QUESTION_CHARS:
        raise HTTPException(400, "That question is too long. Please ask it in fewer words.")
    try:
        clip = _save_clip(audio) if audio is not None else None
    except BadClip as error:
        return StreamingResponse(iter([json.dumps({"event": "error", "message": str(error)}).encode() + b"\n"]))
    if clip is None and not text:
        raise HTTPException(400, "Nothing was asked.")
    conversation = asked_of.chat(chat)

    def work(emit: Callable[[dict], None]) -> None:
        try:
            if not limiter.allow(caller):
                limit = Message("assistant", phrases(conversation.session.language or language)["limit"])
                conversation.messages += [Message("user", text), limit] if text else [limit]
                emit({"event": "reply", "chat": {"id": conversation.id, "title": conversation.title}, "message": _shown(limit)})
                return
            if action:
                emit({"event": "heard", "text": text})
            if clip:
                log.info("voice question received")
            try:
                turn = answer_turn(
                    conversation.session,
                    audio_path=clip,
                    text=None if clip else text,
                    hint=language,
                    on_heard=lambda heard: emit({"event": "heard", "text": heard}),
                )
            except BadClip as error:
                emit({"event": "error", "message": str(error)})     # nothing was asked, so nothing is kept
                return
        finally:
            if clip:
                Path(clip).unlink(missing_ok=True)      # audio is not stored
        reply = Message("assistant", turn.reply_text, turn=turn)
        conversation.messages += [Message("user", turn.transcript, spoken=clip is not None), reply]
        del conversation.messages[:-MAX_MESSAGES]
        conversation.title = conversation.title or _title(turn)
        emit({
            "event": "reply",
            "chat": {"id": conversation.id, "title": conversation.title},
            "message": _shown(reply),
            "details": _details(turn),
            "speak": clip is not None,      # a spoken question is answered aloud; a typed one when the student asks
        })  # fmt: skip

    return StreamingResponse(_events(work), media_type="application/x-ndjson")


# --- The voice ------------------------------------------------------------------

def _voice_refused(caller: str, turn: Turn) -> str | None:
    """Count this reply against the voice limits. Returns why it may not be spoken, or None if it may."""
    # Charged for the text that is actually sent: numbers are spelt out for speech, which can make it far longer.
    rupees = len(tts.clean_for_speech(turn.reply_text, turn.language)[: tts.MAX_TEXT_CHARS]) * budget.VOICE_RUPEES_PER_CHAR
    paise = math.ceil(rupees * 100)
    # Looked at first and only counted once the reply will really be spoken: a refusal must not use up an allowance.
    if not voice_limiter.room(caller):
        return VOICE_LIMIT
    if not visitor_voice.room(caller, paise):
        return VOICE_SHARE
    if not budget.daily.spend(rupees, "voice"):
        return VOICE_BUDGET
    voice_limiter.allow(caller)
    visitor_voice.allow(caller, cost=paise)
    return None


def _unspoken(turn: Turn, message: str, status: int) -> JSONResponse:
    """The reply could not be spoken here. The page is given the words to say with the browser's own voice, if it has one."""
    words = {"text": tts.clean_for_speech(turn.reply_text, turn.language), "language": turn.language}
    return JSONResponse({"message": message, "speech": words, "details": _details(turn)}, status_code=status)


def _message(visitor: Visitor, chat_id: str, key: str) -> Message:
    chat = visitor.find_chat(chat_id)
    message = next((m for m in chat.messages if m.key == key), None) if chat else None
    if message is None or message.turn is None:
        raise HTTPException(404, "That reply is no longer here.")
    return message


@app.post("/api/speak")
def speak_reply(request: Request, chat: str = Body(...), key: str = Body(...), voice: str = Body("")):
    """Speak one reply with the chosen speaker and say where the audio is."""
    visitor = _visitor(request)
    turn = _message(visitor, chat, key).turn
    voice = voice if voice in tts.VOICES else tts.DEFAULT_VOICE
    made = turn.spoken.get(voice)
    if not (made and Path(made).exists()):      # already spoken in this voice: the same file is played again
        # A failed turn plays a recorded "sorry" clip, which costs nothing and is never capped.
        if not turn.error and (refused := _voice_refused(_caller(request, visitor), turn)):
            log.info("the reply is shown but not spoken: %s", refused)
            return _unspoken(turn, refused, 429)
        made = speak(turn, voice)
        if not made:
            return _unspoken(turn, VOICE_FAILED, 503)
        turn.spoken[voice] = made
    return {"url": f"/api/audio/{Path(made).name}", "details": _details(turn)}


@app.get("/api/audio/{name}")
def audio(request: Request, name: str) -> FileResponse:
    """A spoken reply. Only the visitor it was made for can fetch it; the recorded "sorry" clips are open to all."""
    if not _AUDIO_NAME.fullmatch(name):
        raise HTTPException(404)
    if name.startswith("sorry_"):
        path = ASSETS_DIR / name
    else:
        visitor = _visitor(request)
        path = OUT_DIR / name
        turns = (m.turn for chat in visitor.chats() for m in chat.messages if m.turn)
        if not any(str(path) in turn.spoken.values() for turn in turns):
            raise HTTPException(404)
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/feedback")
def feedback(request: Request, chat: str = Body(...), key: str = Body(...), liked: bool = Body(...)) -> dict:
    """Thumbs up or down on a reply, stored with that turn in the log."""
    message = _message(_visitor(request), chat, key)
    saved = message.turn.id is not None and set_feedback(message.turn.id, 1 if liked else -1)
    if saved:
        message.liked = liked
    return {"saved": bool(saved)}


# --- Background jobs and start-up -----------------------------------------------------

def _warm_up() -> None:
    """Load the embedding model once the page is up, so the first question is not slow."""
    try:
        embed_query("warm up")
    except Exception:
        log.exception("could not load the embedding model")


def _purge_forever() -> None:
    """Hourly: delete uploads past the retention limit, and forget visitors who have not been back."""
    while True:
        try:
            purge_expired()
            purge_empty_notebooks(NOTES_TTL_HOURS)
            purge_old_turns(TURN_LOG_DAYS)
        except psycopg.Error:
            log.exception("could not purge expired notes and old turns")
        cutoff = time.time() - NOTES_TTL_HOURS * 3600
        tts.clear_old_replies()
        # Each uploaded file by its own age: a folder's date changes whenever another file is added to it.
        for kept in UPLOADS_DIR.glob("*/*") if UPLOADS_DIR.exists() else []:
            try:
                if kept.stat().st_mtime < cutoff:
                    kept.unlink()
                    if not any(kept.parent.iterdir()):
                        kept.parent.rmdir()
            except OSError:
                log.warning("could not remove an old upload")
        with _visitors_lock:
            for visitor_id in [key for key, visitor in visitors.items() if visitor.seen < cutoff]:
                del visitors[visitor_id]
        time.sleep(3600)


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    # One copy only. A second copy would load the embedding model again and keep usage caps of its own,
    # so refuse to start rather than quietly moving to the next free port.
    if _port_in_use(PORT):
        sys.exit(f"Port {PORT} is busy: the app is already running. Close that window first, then start again.")
    # proxy_headers=False: left on, uvicorn would itself believe an X-Forwarded-For sent from this machine
    # (a tunnel, say) and report that as the caller's address. _address() decides what to trust.
    uvicorn.run(app, host=HOST, port=PORT, log_config=None, proxy_headers=False)


if __name__ == "__main__":
    main()
