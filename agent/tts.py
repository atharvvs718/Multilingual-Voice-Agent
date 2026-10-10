"""Step 9: text to speech, with Sarvam Bulbul v3 over the network.

The student picks one of five speakers in the app. The voice is made on Sarvam's servers,
so nothing runs on this machine's GPU; a reply costs about one rupee.

synthesize() is the whole interface: one reply in, one .wav file out.
"""
import base64
import io
import logging
import re
import time
import uuid

import numpy as np
import soundfile as sf

from agent import sarvam
from agent.lang import MATH
from agent.numbers import spoken_form
from config import ASSETS_DIR, OUT_DIR, SARVAM_TTS_MODEL, TTS_VOICE

log = logging.getLogger(__name__)

# Speaker name (as Sarvam knows it) -> what the student sees in the voice list.
VOICES = {
    "priya": "Priya (female)",
    "pooja": "Pooja (female)",
    "neha": "Neha (female)",
    "shubh": "Shubh (male)",
    "aditya": "Aditya (male)",
}
DEFAULT_VOICE = TTS_VOICE if TTS_VOICE in VOICES else "priya"

SARVAM_LANG = {"mr": "mr-IN", "hi": "hi-IN", "en": "en-IN"}
MAX_TEXT_CHARS = 2500       # the most bulbul:v3 accepts in one request

REPLY_MAX_AGE_S = 600       # spoken replies are not kept: files older than this are deleted
TARGET_RMS = 0.08           # every reply is brought to this average level, so no voice is louder than another
PEAK_LIMIT = 0.95

_SYMBOL = re.compile(r"[\\$()\[\]\s]*([A-Za-z0-9]{1,3})[\\$()\[\]\s]*")      # a formula that is one bare symbol: $G$
_UNSPEAKABLE = re.compile(r"[*_#`~|<>\[\](){}\"“”‘’\\^=]")
_DASHES = re.compile(r"\s*[‐-―/]\s*|(?<=\S)-(?=\S)")


def available_voices() -> list[tuple[str, str]]:
    """(label, speaker name) for the voice list in the app."""
    return [(label, speaker) for speaker, label in VOICES.items()]


# --- Text and audio ---------------------------------------------------------------

def clean_for_speech(text: str, lang: str) -> str:
    """Make a written reply speakable: no markdown or brackets, dashes as pauses, numbers as words.

    A formula is shown, not read: the reply says the relation in words next to it. A lone
    symbol such as $G$ is kept, so "where G is the constant" still makes sense when heard.
    The text always ends with a full stop, so the voice finishes the last sentence cleanly.
    """
    text = MATH.sub(lambda found: f" {symbol.group(1)} " if (symbol := _SYMBOL.fullmatch(found.group())) else " ", text)
    text = _UNSPEAKABLE.sub("", text)
    text = _DASHES.sub(" ", text)
    text = spoken_form(" ".join(text.split()), lang)
    return text if not text or text[-1] in "।.?!" else f"{text}."


def level(samples: np.ndarray) -> np.ndarray:
    """Scale samples to the standard loudness, without clipping."""
    samples = np.atleast_1d(samples).astype(np.float32)
    if samples.size == 0:
        return samples
    rms = float(np.sqrt(np.mean(samples**2)))
    peak = float(np.abs(samples).max())
    return samples * min(TARGET_RMS / rms, PEAK_LIMIT / peak) if rms > 1e-4 else samples


def _decode(audio: bytes) -> tuple[np.ndarray, int]:
    """Samples and sample rate from an encoded audio file, mixed down to one channel."""
    samples, rate = sf.read(io.BytesIO(audio), dtype="float32")
    return (samples.mean(axis=1) if samples.ndim > 1 else samples), rate


def clear_old_replies() -> None:
    """Delete spoken replies that have had time to be played."""
    cutoff = time.time() - REPLY_MAX_AGE_S
    for old in OUT_DIR.glob("reply_*.wav"):
        try:
            if old.stat().st_mtime < cutoff:
                old.unlink()
        except OSError:
            pass    # another request removed it first, or it is still being served


def _new_path() -> str:
    """A fresh file name in OUT_DIR, clearing out replies that have already been played."""
    clear_old_replies()
    return str(OUT_DIR / f"reply_{uuid.uuid4().hex[:8]}.wav")


def fallback_audio(lang: str) -> str | None:
    """Pre-recorded "sorry, please try again" for lang, if scripts/make_fallback_audio.py has been run."""
    path = ASSETS_DIR / f"sorry_{lang}.wav"
    return str(path) if path.exists() else None


# --- Sarvam -----------------------------------------------------------------------

def synthesize(text: str, lang: str, voice: str = DEFAULT_VOICE) -> str:
    """Speak a whole reply in lang ('mr', 'hi' or 'en') with the chosen speaker; returns the path of one .wav file."""
    speaker = voice if voice in VOICES else DEFAULT_VOICE
    lang = lang if lang in SARVAM_LANG else "hi"
    spoken = clean_for_speech(text, lang)[:MAX_TEXT_CHARS]
    answer = sarvam.post(
        "text-to-speech",
        json={"text": spoken, "target_language_code": SARVAM_LANG[lang], "speaker": speaker, "model": SARVAM_TTS_MODEL},
    )
    samples, rate = _decode(base64.b64decode(answer["audios"][0]))
    path = _new_path()
    sf.write(path, level(samples), rate)
    return path
