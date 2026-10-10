"""Step 3: speech to text.

Sarvam Saaras does the listening: it is trained on Indian speech and writes Marathi and Hindi
correctly where Whisper garbles them. Groq Whisper is the backup when Sarvam cannot be used.
"""
import logging

import groq
import numpy as np

from agent import budget, sarvam
from agent.audio import to_wav_bytes
from agent.errors import ServiceUnavailable
from agent.llm import TRANSIENT, groq_client
from agent.retry import with_retries
from config import SAMPLE_RATE, SARVAM_STT_MODEL, STT_MODEL

log = logging.getLogger(__name__)

# "codemix" writes English words in Latin letters and Indian words in their own script,
# so a term such as "gravitation" reaches the notes search spelt the way a textbook spells it.
SARVAM_MODE = "codemix"


def _sarvam(audio: bytes, hint: str | None) -> tuple[str, str]:
    answer = sarvam.post(
        "speech-to-text",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data={"model": SARVAM_STT_MODEL, "mode": SARVAM_MODE, "language_code": f"{hint}-IN" if hint else "unknown"},
    )
    language = (answer.get("language_code") or "").split("-")[0].lower()      # "mr-IN" -> "mr"
    return (answer.get("transcript") or "").strip(), language


def _whisper(audio: bytes, hint: str | None) -> tuple[str, str]:
    kwargs = {"language": hint} if hint else {}

    def request():
        return groq_client().audio.transcriptions.create(
            file=("clip.wav", audio),
            model=STT_MODEL,
            response_format="verbose_json",
            temperature=0.0,
            **kwargs,
        )

    try:
        result = with_retries(request, (*TRANSIENT, groq.RateLimitError))
    except (*TRANSIENT, groq.RateLimitError) as error:
        raise ServiceUnavailable(f"STT unavailable: {type(error).__name__}") from error
    return result.text.strip(), (getattr(result, "language", "") or "").lower()


def transcribe(wav: np.ndarray, hint: str | None = None) -> tuple[str, str]:
    """Return (transcript, language the speech model heard). hint is 'mr', 'hi' or None for auto."""
    audio = to_wav_bytes(wav)
    cost = len(wav) / SAMPLE_RATE * budget.LISTENING_RUPEES_PER_SECOND
    # Past the day's budget Whisper listens instead: free, and worse at Marathi.
    if sarvam.has_key() and budget.daily.spend(cost, "listening"):
        try:
            return _sarvam(audio, hint)
        except ServiceUnavailable as error:
            log.warning("Sarvam could not transcribe (%s); using Whisper instead", error)
    return _whisper(audio, hint)
