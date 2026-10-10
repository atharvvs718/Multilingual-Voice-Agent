"""Exceptions the pipeline treats specially."""


class BadClip(Exception):
    """Clip is empty, too short or too long. The message is safe to show the user."""


class ServiceUnavailable(Exception):
    """An external API (STT, LLM, TTS) kept failing after retries and fallbacks."""
