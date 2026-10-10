"""Speech to text: the Sarvam request, and Whisper taking over when Sarvam cannot be used. No network."""
from types import SimpleNamespace

import numpy as np
import pytest

import agent.sarvam as sarvam_api
import agent.stt as stt

CLIP = np.zeros(1600, dtype=np.float32)


class FakeResponse:
    """Shaped like a requests response from Sarvam's speech-to-text."""

    def __init__(self, status_code: int = 200, transcript: str = "", language_code: str | None = "mr-IN", text: str = ""):
        self.status_code, self.text, self.ok = status_code, text, status_code < 400
        self._body = {"request_id": "r1", "transcript": transcript, "language_code": language_code}

    def json(self):
        return self._body


@pytest.fixture
def sarvam(monkeypatch):
    """A stand-in for Sarvam: records each request and answers with the responses given, in order."""
    sent = []

    def install(*responses):
        queue = list(responses)

        def fake_post(url, headers, timeout, files, data):
            sent.append({"url": url, "file": files["file"][0], "data": data})
            return queue.pop(0)

        monkeypatch.setattr(sarvam_api.requests, "post", fake_post)
        return sent

    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    return install


@pytest.fixture
def whisper(monkeypatch):
    """A stand-in for Groq Whisper: records the arguments of each call."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=" प्रकाशसंश्लेषण म्हणजे काय? ", language="Marathi")

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    monkeypatch.setattr(stt, "groq_client", lambda: client)
    return calls


def test_sarvam_is_asked_in_code_mix_mode_with_the_students_language(sarvam, whisper):
    sent = sarvam(FakeResponse(transcript=" Gravitation म्हणजे काय? पाच वाक्यात सांगा. "))
    assert stt.transcribe(CLIP, "mr") == ("Gravitation म्हणजे काय? पाच वाक्यात सांगा.", "mr")
    assert sent == [
        {
            "url": "https://api.sarvam.ai/speech-to-text",
            "file": "clip.wav",
            "data": {"model": stt.SARVAM_STT_MODEL, "mode": "codemix", "language_code": "mr-IN"},
        }
    ]
    assert whisper == []


def test_without_a_hint_sarvam_detects_the_language(sarvam, whisper):
    sent = sarvam(FakeResponse(transcript="प्रकाश संश्लेषण क्या है?", language_code="hi-IN"))
    assert stt.transcribe(CLIP) == ("प्रकाश संश्लेषण क्या है?", "hi")
    assert sent[0]["data"]["language_code"] == "unknown"


def test_a_missing_language_in_the_answer_is_returned_as_empty(sarvam, whisper):
    sarvam(FakeResponse(transcript="ऑक्सिजन", language_code=None))
    assert stt.transcribe(CLIP) == ("ऑक्सिजन", "")


def test_whisper_takes_over_when_sarvam_refuses(sarvam, whisper):
    sent = sarvam(FakeResponse(403, text="no credit left"))
    assert stt.transcribe(CLIP, "mr") == ("प्रकाशसंश्लेषण म्हणजे काय?", "marathi")
    assert len(sent) == 1
    assert whisper[0]["language"] == "mr"


def test_whisper_takes_over_when_sarvam_stays_busy(sarvam, whisper):
    sent = sarvam(*[FakeResponse(503)] * 4)
    assert stt.transcribe(CLIP, "mr")[0] == "प्रकाशसंश्लेषण म्हणजे काय?"
    assert len(sent) == 4       # the first try and three retries


def test_without_a_sarvam_key_whisper_is_used_directly(sarvam, whisper, monkeypatch):
    sent = sarvam()
    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "")
    assert stt.transcribe(CLIP) == ("प्रकाशसंश्लेषण म्हणजे काय?", "marathi")
    assert sent == []
    assert "language" not in whisper[0]
