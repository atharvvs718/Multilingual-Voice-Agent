"""Step 1-2 of a turn: load any audio file as 16 kHz mono, trim silence, reject bad clips."""
import io
import logging
import subprocess
from functools import lru_cache

import numpy as np
import soundfile as sf

from agent.errors import BadClip
from config import CHILD_ENV, MAX_CLIP_S, MIN_CLIP_S, SAMPLE_RATE

log = logging.getLogger(__name__)
DECODE_TIMEOUT_S = 30
UNREADABLE = "Could not read the audio file. Please try again."


def load_16k_mono(path: str) -> np.ndarray:
    """Decode .wav / .mp3 / browser recordings with ffmpeg into float32 samples."""
    # -t stops the decoder a little past the longest clip the app accepts: a small file can claim to be hours
    # long, and the whole decoded clip is held in memory. The length check in preprocess() then refuses it.
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(MAX_CLIP_S + 5), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=DECODE_TIMEOUT_S, env=CHILD_ENV)
    except subprocess.TimeoutExpired as error:
        raise BadClip(UNREADABLE) from error
    if proc.returncode != 0:
        log.warning("ffmpeg could not read a clip: %s", proc.stderr.decode(errors="ignore").strip()[:300])
        raise BadClip(UNREADABLE)       # what ffmpeg said names files on this machine; it stays in the log
    return np.frombuffer(proc.stdout, dtype=np.float32)


@lru_cache(maxsize=1)
def _vad_model():
    from silero_vad import load_silero_vad

    return load_silero_vad()


def trim_silence(wav: np.ndarray, pad_s: float = 0.2) -> np.ndarray:
    """Keep audio from the first speech sample to the last, with a little padding."""
    import torch
    from silero_vad import get_speech_timestamps

    stamps = get_speech_timestamps(torch.from_numpy(wav.copy()), _vad_model(), sampling_rate=SAMPLE_RATE)
    if not stamps:
        return wav[:0]
    pad = int(pad_s * SAMPLE_RATE)
    start = max(0, stamps[0]["start"] - pad)
    end = min(len(wav), stamps[-1]["end"] + pad)
    return wav[start:end]


def preprocess(path: str) -> np.ndarray:
    """Load, trim and length-check one clip. Raises BadClip if it should not be sent to STT."""
    wav = trim_silence(load_16k_mono(path))
    seconds = len(wav) / SAMPLE_RATE
    if seconds < MIN_CLIP_S:
        raise BadClip("No speech heard. Please try again.")
    if seconds > MAX_CLIP_S:
        raise BadClip(f"Clip is {seconds:.0f} s; keep it under {MAX_CLIP_S:.0f} s.")
    return wav


def to_wav_bytes(wav: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()
