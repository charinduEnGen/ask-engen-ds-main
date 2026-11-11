"""Executes agent-1-generated SQL against a project's SQLite database.

Moved from src/tools/sql_query_tool.py; that file is now a backward-compat
re-export.
"""

import re

import pandas as pd
from sqlalchemy.engine import Engine

_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)


def execute_sql_query(sql: str, engine: Engine):
    """Returns a DataFrame on success, or a "TOOL_ERROR: ..." string on failure."""
    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].strip()

    if not statement:
        return "TOOL_ERROR: empty SQL statement"
    if ";" in statement:
        return "TOOL_ERROR: only a single SQL statement is allowed"
    if not statement.upper().startswith(("SELECT", "WITH")):
        return "TOOL_ERROR: only a single read-only SELECT statement is allowed"
    if _FORBIDDEN_RE.search(statement):
        return "TOOL_ERROR: statement contains a disallowed keyword — only read-only SELECT queries are permitted"

    try:
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA query_only = ON")
            result = pd.read_sql_query(statement, conn)
    except Exception as exc:  # noqa: BLE001
        return f"TOOL_ERROR: {type(exc).__name__}: {exc}"

    return result.reset_index(drop=True)
