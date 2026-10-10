"""Create the tables in whatever database DATABASE_URL points at.

Docker runs schema.sql by itself on first start. A hosted database (Neon) does not,
so run this once after setting DATABASE_URL:  python -m db.init
"""
from pathlib import Path

from db.conn import connect

SCHEMA = Path(__file__).with_name("schema.sql")


def init() -> None:
    connect().execute(SCHEMA.read_text(encoding="utf-8"))


if __name__ == "__main__":
    init()
    print("Schema applied.")
