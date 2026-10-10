"""One daily budget, in rupees, for everything that costs money: Sarvam's voice and Sarvam's listening.

Every paid call asks spend() first. Once the day's budget is used the answer is no, and the
caller falls back to something free: Whisper for listening, the browser's own voice or plain
text for speaking.

The count lives in the database and nowhere else: each charge is one statement that raises
the day's total only if it stays within the budget. So a restart, or a second copy of the
app, sees the same total. If the database cannot be reached the answer is also no:
guessing would hand out a fresh budget every time it happened.
"""
import logging
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import psycopg

from config import DAILY_BUDGET_RUPEES
from db.conn import run

log = logging.getLogger(__name__)

# Sarvam's prices: Bulbul v3 is Rs 30 for 10,000 characters, Saaras is Rs 30 for an hour of audio.
VOICE_RUPEES_PER_CHAR = 30 / 10_000
LISTENING_RUPEES_PER_SECOND = 30 / 3600
_INDIA = timedelta(hours=5, minutes=30)
TOTAL = "total"     # the row that holds the day's total; the other rows say what it was spent on
# The total is checked and raised in one step, on a row the database locks while it does so,
# so two copies of the app charging at the same moment cannot both squeeze under the limit.
# Only if that succeeded is the cost also added to its own service's row.
_CHARGE = """
WITH counted AS (
    INSERT INTO spend (day, service, rupees)
    SELECT %(day)s, 'total', %(rupees)s WHERE %(rupees)s <= %(limit)s
    ON CONFLICT (day, service) DO UPDATE SET rupees = spend.rupees + excluded.rupees
        WHERE spend.rupees + excluded.rupees <= %(limit)s
    RETURNING rupees
)
INSERT INTO spend (day, service, rupees)
SELECT %(day)s, %(service)s, %(rupees)s FROM counted
ON CONFLICT (day, service) DO UPDATE SET rupees = spend.rupees + excluded.rupees
RETURNING rupees
"""


def _today() -> date:
    """The date in India, where the students are: the budget starts again at their midnight."""
    return (datetime.now(UTC) + _INDIA).date()


def _charge(day: date, service: str, rupees: float, limit: float) -> bool:
    """Add a cost to the day's total if it still fits; True if it was added."""
    return bool(run(_CHARGE, {"day": day, "service": service, "rupees": rupees, "limit": limit}))


def _spent(day: date) -> float:
    return float(run("SELECT coalesce(sum(rupees), 0) FROM spend WHERE day = %s AND service = %s", (day, TOTAL))[0][0])


class Budget:
    """At most rupees_per_day spent in one day."""

    def __init__(
        self,
        rupees_per_day: float,
        charge: Callable[[date, str, float, float], bool] = _charge,
        spent: Callable[[date], float] = _spent,
        today: Callable[[], date] = _today,
    ):
        self.rupees_per_day = rupees_per_day
        self._charge, self._spent, self._today = charge, spent, today

    def spend(self, rupees: float, service: str) -> bool:
        """Record a cost and return True, or return False if it would take today over the budget."""
        if not math.isfinite(rupees) or rupees < 0:
            return False
        try:
            return self._charge(self._today(), service, rupees, self.rupees_per_day)
        except psycopg.Error:
            log.warning("the day's spending could not be checked, so nothing paid is used until the database is back")
            return False

    def spent(self) -> float:
        """What has been spent today; the whole budget if that cannot be read."""
        try:
            return self._spent(self._today())
        except psycopg.Error:
            return self.rupees_per_day


daily = Budget(DAILY_BUDGET_RUPEES)
