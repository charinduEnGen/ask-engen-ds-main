"""Per-project SQLite engine. Each project gets its own file at
data/<project>/<project>.db (created by src.routes.datasets.confirm), queried
live by the SQL query agent — see src.tools.sql_query_tool. No single
app-wide database: engines are created lazily, one per project, and cached
for the lifetime of the process.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import DATA_DIR

_engines: dict[str, Engine] = {}


def db_path(project: str) -> Path:
    return DATA_DIR / project / f"{project}.db"


def get_engine(project: str) -> Engine:
    if project not in _engines:
        path = db_path(project)
        path.parent.mkdir(parents=True, exist_ok=True)
        _engines[project] = create_engine(f"sqlite:///{path}", future=True)
    return _engines[project]


def reset_engine(project: str) -> None:
    """Disposes and drops the cached engine for `project` — call before
    recreating its SQLite file on a re-confirm so no stale pooled connection
    keeps the old file open or cached."""
    engine = _engines.pop(project, None)
    if engine is not None:
        engine.dispose()


def has_project_db(project: str) -> bool:
    return db_path(project).exists()
