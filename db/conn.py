"""PostgreSQL access. One shared connection, reopened automatically if it drops.

A hosted database such as Neon closes idle connections, so every statement gets one
reconnect-and-retry before the error is allowed to surface.
"""
import threading
import time
from collections.abc import Sequence

import psycopg
from pgvector.psycopg import register_vector

from config import DATABASE_URL

CONNECT_TIMEOUT_S = 3
RETRY_AFTER_S = 15      # after a failed connect, fail fast for this long instead of making every call wait

_lock = threading.Lock()
_connection: psycopg.Connection | None = None
_down_until = 0.0


def connect() -> psycopg.Connection:
    """Return the shared connection, opening it if needed. psycopg serialises use across threads."""
    global _connection, _down_until
    with _lock:
        if _connection is None or _connection.closed or _connection.broken:
            if time.monotonic() < _down_until:
                raise psycopg.OperationalError("database is unreachable (checked a moment ago)")
            try:
                _connection = psycopg.connect(DATABASE_URL, autocommit=True, connect_timeout=CONNECT_TIMEOUT_S)
            except psycopg.OperationalError:
                _down_until = time.monotonic() + RETRY_AFTER_S
                raise
            register_vector(_connection)
        return _connection


def _reset() -> None:
    global _connection
    with _lock:
        if _connection is not None and not _connection.closed:
            _connection.close()
        _connection = None


def run(sql: str, params: Sequence | None = None) -> list[tuple]:
    """Execute one statement and return its rows (an empty list for statements that return none)."""
    for attempt in (1, 2):
        connection = connect()      # a database that is down fails here, once, without a retry
        try:
            cursor = connection.execute(sql, params)
            return cursor.fetchall() if cursor.description else []
        except psycopg.OperationalError:
            # The connection had gone stale (a hosted database closes idle ones): reopen and try once more.
            _reset()
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def run_many(sql: str, rows: Sequence[Sequence]) -> None:
    """Execute one statement once per row, in a single round trip where the driver allows."""
    for attempt in (1, 2):
        connection = connect()
        try:
            with connection.cursor() as cursor:
                cursor.executemany(sql, rows)
            return
        except psycopg.OperationalError:
            _reset()
            if attempt == 2:
                raise
