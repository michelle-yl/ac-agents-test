"""PostgreSQL access for monitoring replica."""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Generator

import psycopg

from sdl_agents.config import DB_TIMEOUT_SECONDS, MAX_RESULT_ROWS, database_url

WRITE_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|VACUUM|ANALYZE|EXECUTE|CALL)\b",
    re.IGNORECASE,
)


def assert_read_only_sql(sql: str) -> None:
    stripped = sql.strip()
    if not stripped.upper().startswith("SELECT"):
        raise ValueError("Only SELECT statements are allowed")
    if WRITE_KEYWORDS.search(stripped):
        raise ValueError("Write operations are not allowed")


@contextmanager
def connect() -> Generator[psycopg.Connection, None, None]:
    with psycopg.connect(database_url(), connect_timeout=int(DB_TIMEOUT_SECONDS)) as conn:
        yield conn


def fetch_all(sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    assert_read_only_sql(sql)
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{int(DB_TIMEOUT_SECONDS * 1000)}'")
            cur.execute(sql, params or ())
            if cur.description is None:
                return []
            columns = [d.name for d in cur.description]
            rows = cur.fetchmany(MAX_RESULT_ROWS + 1)
            if len(rows) > MAX_RESULT_ROWS:
                raise ValueError(f"Result exceeds MAX_RESULT_ROWS ({MAX_RESULT_ROWS})")
            return [dict(zip(columns, row)) for row in rows]
