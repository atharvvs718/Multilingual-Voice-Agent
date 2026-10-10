"""All settings in one place, read from .env. Switching a model is a change here, not in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

# --- Keys -------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# --- Models -----------------------------------------------------------------
# Sarvam listens and speaks. Whisper (on Groq) is the backup listener when Sarvam cannot be used.
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v4")
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
# A model that can read a picture: it looks at a page of the book when the student asks about an example on it.
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
# The voice selected when the app opens; the student can change it in the Studio.
# One of the speakers listed in agent.tts.VOICES: priya, pooja, neha, shubh, aditya.
TTS_VOICE = os.getenv("TTS_VOICE", "priya")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
API_TIMEOUT_S = float(os.getenv("API_TIMEOUT_S", "15"))

# --- Database ---------------------------------------------------------------
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and stalls.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://voice:voice@127.0.0.1:5432/voice")

# --- Audio ------------------------------------------------------------------
SAMPLE_RATE = 16000
MIN_CLIP_S = 0.5
MAX_CLIP_S = 30.0

# --- Conversation and limits ------------------------------------------------
MAX_HISTORY = 8                 # user + assistant messages sent with every call; older ones are summarised
MAX_TURNS_PER_HOUR = int(os.getenv("MAX_TURNS_PER_HOUR", "30"))
# The voice is the paid part of a turn (about 90% of its cost), so it has a cap of its own.
MAX_SPOKEN_PER_HOUR = int(os.getenv("MAX_SPOKEN_PER_HOUR", "20"))              # spoken replies, per visitor
MAX_UPLOAD_MB = 25              # a whole textbook PDF is commonly 10-20 MB
MAX_CHUNKS_PER_SESSION = 2000   # roughly a 400-page book; stops one visitor filling the database
NOTES_TTL_HOURS = 24            # uploads are deleted when the session ends, and after this at the latest
# The most the paid services (voice and listening) may cost in one day, all visitors together. Past it the
# app keeps working on the free fallbacks. See agent/budget.py.
DAILY_BUDGET_RUPEES = float(os.getenv("DAILY_BUDGET_RUPEES", "30"))
MAX_UPLOADS_PER_HOUR = int(os.getenv("MAX_UPLOADS_PER_HOUR", "6"))      # per visitor: indexing a book is the heaviest thing the app does
TURN_LOG_DAYS = int(os.getenv("TURN_LOG_DAYS", "7"))                    # questions and replies are deleted from the log after this
# If set, the app asks for this code before it can be used: for a demo shared with a few people.
ACCESS_CODE = os.getenv("ACCESS_CODE", "")
MAX_QUESTION_CHARS = 1000       # a typed question; longer ones are refused before any model sees them
# How many proxies stand between the internet and this app (0 on a laptop, 1 on a hosting service).
# With 0 the X-Forwarded-For header is ignored, because anyone can send it; with 1 the last entry is
# used, which is the one the hosting service itself wrote.
PROXY_HOPS = int(os.getenv("PROXY_HOPS", "0"))
HOST = os.getenv("HOST", "127.0.0.1")       # 0.0.0.0 inside a container
PORT = int(os.getenv("PORT", "7860"))

# --- Paths ------------------------------------------------------------------
OUT_DIR = ROOT / "out"          # spoken replies; old files are removed as new ones are written
ASSETS_DIR = ROOT / "assets"    # pre-recorded fallback audio
NOTES_DIR = ROOT / "notes"      # starter notes that ship with the app
UPLOADS_DIR = ROOT / "uploads"  # each visitor's uploaded files, kept so their pages can be looked at; deleted with the notes
OUT_DIR.mkdir(exist_ok=True)

# What a helper process is started with: the environment without the secrets. The helpers read
# files that strangers upload (PDFs, audio), so they are given nothing worth stealing.
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL", "ACCESS_CODE")
CHILD_ENV = {name: value for name, value in os.environ.items() if not any(word in name.upper() for word in _SECRET_WORDS)}
