"""Retry with backoff for calls to external APIs (FR-16)."""
import logging
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")
BACKOFF_S = (0.5, 1.0, 2.0)


def with_retries(
    fn: Callable[[], T],
    retry_on: tuple[type[Exception], ...],
    delays: tuple[float, ...] = BACKOFF_S,
) -> T:
    """Call fn; on one of retry_on, wait and try again, up to len(delays) more times.

    The last failure is re-raised unchanged so the caller can decide what to do next.
    """
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return fn()
        except retry_on as error:
            if delay is None:
                raise
            log.warning("attempt %d failed (%s: %s); retrying in %.1f s", attempt, type(error).__name__, error, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")
