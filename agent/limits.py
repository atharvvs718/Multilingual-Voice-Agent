"""Usage caps (FR-16): stop one person, or a busy day, burning the free tier or the paid voice credit."""
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    """Allow at most max_events per key in any window of window_s seconds.

    An event normally counts as one. It can be given a cost instead, which makes the cap a
    budget: the voice is paid for by the character, so a spoken reply costs its length.
    """

    MAX_KEYS = 10_000

    def __init__(self, max_events: int, window_s: float, clock: Callable[[], float] = time.monotonic):
        self.max_events = max_events
        self.window_s = window_s
        self._clock = clock
        self._events: dict[str, deque[tuple[float, int]]] = defaultdict(deque)     # key -> (time, cost)
        self._lock = threading.Lock()

    def allow(self, key: str, cost: int = 1) -> bool:
        """Record one event for key and return True, or return False if it would take key over the cap."""
        now = self._clock()
        with self._lock:
            if len(self._events) > self.MAX_KEYS:
                self._forget_idle(now)
            events = self._events[key]
            while events and now - events[0][0] >= self.window_s:
                events.popleft()
            if sum(spent for _, spent in events) + cost > self.max_events:
                return False
            events.append((now, cost))
            return True

    def room(self, key: str, cost: int = 1) -> bool:
        """Would allow() succeed now? Nothing is recorded, so a check that is followed by a refusal elsewhere costs nothing."""
        now = self._clock()
        with self._lock:
            events = self._events.get(key, ())
            return sum(spent for at, spent in events if now - at < self.window_s) + cost <= self.max_events

    def _forget_idle(self, now: float) -> None:
        """Drop visitors with nothing inside the window, so the table cannot grow without bound."""
        for key in [k for k, events in self._events.items() if not events or now - events[-1][0] >= self.window_s]:
            del self._events[key]
