import numpy as np
import pytest
import soundfile as sf

from agent.audio import preprocess, to_wav_bytes
from agent.errors import BadClip
from config import SAMPLE_RATE


def test_silence_is_rejected(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SAMPLE_RATE * 2, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(BadClip):
        preprocess(str(path))


def test_unreadable_file_is_rejected(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_text("this is not audio")
    with pytest.raises(BadClip):
        preprocess(str(path))


def test_wav_bytes_round_trip():
    samples = np.linspace(-0.5, 0.5, SAMPLE_RATE, dtype=np.float32)
    data = to_wav_bytes(samples)
    assert data[:4] == b"RIFF"
    assert len(data) < samples.nbytes     # 16-bit on the wire, half the size of float32
