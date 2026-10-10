"""Text preparation, loudness and the Sarvam request. Nothing here calls Sarvam."""
import base64
import io

import numpy as np
import pytest
import soundfile as sf

import agent.sarvam as sarvam_api
import agent.tts as tts
from agent.errors import ServiceUnavailable
from agent.tts import clean_for_speech, level

TONE = np.sin(np.linspace(0, 400, 8000)).astype(np.float32)


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))


class FakeResponse:
    """Shaped like a requests response from Sarvam."""

    def __init__(self, status_code: int = 200, text: str = "", stereo: bool = False):
        self.status_code, self.text, self.ok = status_code, text, status_code < 400
        encoded = io.BytesIO()
        sf.write(encoded, np.stack([TONE * 0.9] * 2, axis=1) if stereo else TONE * 0.9, 22050, format="WAV")
        self._audio = base64.b64encode(encoded.getvalue()).decode()

    def json(self):
        return {"request_id": "r1", "audios": [self._audio]}


@pytest.fixture
def sarvam(monkeypatch, tmp_path):
    """A stand-in for Sarvam: records each request and answers with the responses given, in order."""
    sent = []

    def install(*responses):
        queue = list(responses)

        def fake_post(url, headers, json, timeout):
            sent.append({"url": url, "headers": headers, "body": json})
            return queue.pop(0)

        monkeypatch.setattr(sarvam_api.requests, "post", fake_post)
        return sent

    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "a-key")
    monkeypatch.setattr(tts, "_new_path", lambda: str(tmp_path / "reply.wav"))
    return install


# --- text ---------------------------------------------------------------------------

def test_clean_for_speech_removes_what_cannot_be_spoken():
    assert clean_for_speech("**स्टार्च** (starch) निळा‑काळा होतो. पान 12.", "mr") == "स्टार्च starch निळा काळा होतो. पान बारा."
    assert clean_for_speech("ग्लुकोज/स्टार्च तयार  होते", "mr") == "ग्लुकोज स्टार्च तयार होते."
    assert clean_for_speech("well-known fact", "en") == "well known fact."


def test_a_formula_that_slips_into_a_reply_is_not_read_out():
    assert clean_for_speech(r"मोमेंट ऑफ इनर्शिया \(I = M K^{2}\) असतो. पान 14.", "mr") == "मोमेंट ऑफ इनर्शिया असतो. पान चौदा."


def test_a_formula_is_shown_not_spoken_but_a_lone_symbol_is_kept():
    reply = "बल वस्तुमान आणि त्वरण यांच्या गुणाकाराइतके असते.\n$$F = m a$$\nयेथे $F$ म्हणजे बल आणि $m$ म्हणजे वस्तुमान. पान 4."
    assert clean_for_speech(reply, "mr") == (
        "बल वस्तुमान आणि त्वरण यांच्या गुणाकाराइतके असते. येथे F म्हणजे बल आणि m म्हणजे वस्तुमान. पान चार."
    )
    assert clean_for_speech(r"ऊर्जा \[E = m c^{2}\] आहे", "mr") == "ऊर्जा आहे."


def test_spoken_text_always_ends_with_a_full_stop():
    assert clean_for_speech("हे पानांमध्ये होते. पान 1", "mr") == "हे पानांमध्ये होते. पान एक."
    assert clean_for_speech("आणखी एक प्रश्न हवा आहे का?", "mr") == "आणखी एक प्रश्न हवा आहे का?"     # already ends a sentence
    assert clean_for_speech("", "mr") == ""


# --- loudness -------------------------------------------------------------------------

def test_quiet_and_loud_audio_end_up_equally_loud():
    quiet, loud = level(TONE * 0.01), level(TONE * 0.9)
    assert abs(rms(quiet) - rms(loud)) < 0.001
    assert np.abs(loud).max() <= tts.PEAK_LIMIT
    assert level(np.zeros(0)).size == 0             # empty input is left alone


# --- voices ---------------------------------------------------------------------------

def test_the_app_offers_exactly_the_five_chosen_voices_with_priya_first():
    assert tts.available_voices() == [
        ("Priya (female)", "priya"),
        ("Pooja (female)", "pooja"),
        ("Neha (female)", "neha"),
        ("Shubh (male)", "shubh"),
        ("Aditya (male)", "aditya"),
    ]
    assert tts.DEFAULT_VOICE == "priya"


# --- the request ----------------------------------------------------------------------

def test_request_matches_sarvams_documented_shape(sarvam):
    sent = sarvam(FakeResponse(stereo=True))
    path = tts.synthesize("**उत्तर** पान 12", "mr", "pooja")
    assert sent == [
        {
            "url": "https://api.sarvam.ai/text-to-speech",
            "headers": {"api-subscription-key": "a-key"},
            "body": {"text": "उत्तर पान बारा.", "target_language_code": "mr-IN", "speaker": "pooja", "model": tts.SARVAM_TTS_MODEL},
        }
    ]
    samples, rate = sf.read(path)
    assert rate == 22050
    assert samples.ndim == 1                                    # mixed down to one channel
    assert abs(rms(samples) - tts.TARGET_RMS) < 0.001           # and brought to the standard loudness


def test_hindi_replies_are_sent_as_hindi_and_unknown_voices_use_the_default(sarvam):
    sent = sarvam(FakeResponse())
    tts.synthesize("नमस्ते", "hi", "no-such-voice")
    assert sent[0]["body"]["target_language_code"] == "hi-IN"
    assert sent[0]["body"]["speaker"] == tts.DEFAULT_VOICE


def test_a_busy_server_is_retried(sarvam):
    sent = sarvam(FakeResponse(503), FakeResponse(429), FakeResponse())
    tts.synthesize("नमस्ते", "mr")
    assert len(sent) == 3


def test_a_refused_request_is_reported_and_not_retried(sarvam):
    sent = sarvam(FakeResponse(403, "invalid api key"))
    with pytest.raises(ServiceUnavailable, match=r"HTTP 403$"):
        tts.synthesize("नमस्ते", "mr")
    assert len(sent) == 1


def test_a_missing_key_is_reported_without_calling_sarvam(sarvam, monkeypatch):
    sent = sarvam()
    monkeypatch.setattr(sarvam_api, "SARVAM_API_KEY", "")
    with pytest.raises(ServiceUnavailable, match="SARVAM_API_KEY is missing"):
        tts.synthesize("नमस्ते", "mr")
    assert sent == []


def test_very_long_text_is_cut_to_what_sarvam_accepts(sarvam):
    sent = sarvam(FakeResponse())
    tts.synthesize("शब्द " * 1000, "mr")
    assert len(sent[0]["body"]["text"]) <= tts.MAX_TEXT_CHARS
