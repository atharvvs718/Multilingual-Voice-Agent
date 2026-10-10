"""Calls to Sarvam, which does both the listening (agent/stt.py) and the speaking (agent/tts.py)."""
import logging

import requests

from agent.errors import ServiceUnavailable
from agent.retry import with_retries
from config import API_TIMEOUT_S, SARVAM_API_KEY

log = logging.getLogger(__name__)
BASE_URL = "https://api.sarvam.ai"


class _Busy(Exception):
    """Sarvam answered 429 or 503: it did not take the request, so sending it again costs nothing."""


# Sent again only when the request was certainly not carried out: Sarvam said it was busy, or the
# connection was never made. A request that reached Sarvam and then failed in any other way (no answer in
# time, a broken connection, another 5xx) may already have been done and charged for, so it is not repeated.
_RETRIED = (requests.ConnectTimeout, _Busy)


def has_key() -> bool:
    return bool(SARVAM_API_KEY)


def post(endpoint: str, **payload) -> dict:
    """POST payload (json=..., or files=... with data=...) to a Sarvam endpoint and return its JSON answer.

    Anything that goes wrong raises ServiceUnavailable naming the kind of failure.
    """
    if not SARVAM_API_KEY:
        raise ServiceUnavailable("SARVAM_API_KEY is missing. Put it in .env")

    def request() -> dict:
        response = requests.post(
            f"{BASE_URL}/{endpoint}", headers={"api-subscription-key": SARVAM_API_KEY}, timeout=API_TIMEOUT_S, **payload
        )
        if response.status_code in (429, 503):
            raise _Busy(f"HTTP {response.status_code}")
        if not response.ok:
            # A wrong key, an unknown speaker, no credit left. What Sarvam said goes to the log, not to the visitor.
            log.warning("Sarvam refused a request: HTTP %s %s", response.status_code, response.text[:200])
            raise ServiceUnavailable(f"Sarvam refused the request: HTTP {response.status_code}")
        return response.json()

    try:
        return with_retries(request, _RETRIED)
    except (requests.RequestException, _Busy) as error:
        raise ServiceUnavailable(f"Sarvam unavailable: {type(error).__name__}") from error
