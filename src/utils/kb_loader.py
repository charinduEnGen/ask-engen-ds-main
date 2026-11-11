"""Loads a project's schema (per table) and data_context/analytics-patterns
(per project, shared across every table in it) from:
  - data/<project>/<table>/schema.json       — per table
  - data/<project>/data_context.json         — per project
  - data/<project>/patterns.json             — per project

Schema stays per-table since each table's columns are its own. Data context
and query patterns are project-level: the query agent reasons about every
table in the project together (joins, cross-table comparisons), so there's
one shared narrative/pattern set rather than one forked per table.

These files are written once by src.routes.datasets.confirm, right after the
schema-discovery agent's proposal is reviewed and accepted by the user, and
read back here on every query turn. There is no "mode" concept anymore — the
project IS the namespace (see src.schemas.user_request.UserRequest.mode,
which carries the project name end-to-end from the backend).
"""

import json
from datetime import datetime
from pathlib import Path

from src.config import DATA_DIR


def _project_dir(project: str) -> Path:
    return DATA_DIR / project


def _table_dirs(project: str) -> list[Path]:
    proj_dir = _project_dir(project)
    if not proj_dir.exists():
        return []
    return sorted(p for p in proj_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_tables(project: str) -> list[str]:
    return [p.name for p in _table_dirs(project)]


def load_schema(project: str) -> dict:
    """Combined schema for every table confirmed under this project:
    {"tables": [{"name": <table>, "columns": [...]}], "relationships": [...]}.
    This whole object is what gets JSON-dumped into the data_analysis_task
    prompt — the SQL agent's authoritative list of what's queryable and how
    tables join together."""
    tables = []
    for table_dir in _table_dirs(project):
        table_schema = _load_json(table_dir / "schema.json")
        if table_schema:
            tables.append({"name": table_dir.name, "columns": table_schema.get("columns", [])})
    relationships = _load_json(_project_dir(project) / "relationships.json").get("relationships", [])
    return {"tables": tables, "relationships": relationships}


def load_analytics_patterns(project: str) -> list[dict]:
    """The project-level list of reusable query patterns proposed by the
    schema-discovery agent and accepted at confirm time — shared across
    every table in the project, since a pattern may join more than one of
    them (see each pattern's "tables_involved"/"join_hint")."""
    return _load_json(_project_dir(project) / "patterns.json").get("patterns", [])


def load_data_context(project: str) -> dict:
    """{description, currency, time_grain, notes, tables: {<table>:
    {row_count, source_file, sheet}}, last_refreshed} — written once at
    confirm time by src.routes.datasets.confirm."""
    return _load_json(_project_dir(project) / "data_context.json")


def render_prompt_summary(project: str) -> str:
    """The compact, human-readable data-context block fed to the agents'
    prompts: the project-level narrative plus one line per table."""
    context = load_data_context(project)
    if not context:
        return f"No tables have been confirmed yet for project '{project}'."

    lines = [f"Project: {project}"]
    description = context.get("description")
    if description:
        lines.append(description)
    currency = context.get("currency")
    if currency:
        lines.append(f"Currency: {currency}")
    time_grain = context.get("time_grain")
    if time_grain and time_grain != "unknown":
        lines.append(f"Time grain: {time_grain}")
    notes = context.get("notes") or []
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in notes)

    tables = context.get("tables") or {}
    if tables:
        lines.append("Tables:")
        for table, facts in tables.items():
            row_count = facts.get("row_count", "unknown")
            source_file = facts.get("source_file", "")
            lines.append(f"- {table}: {row_count} row(s), from {source_file}".strip())
    return "\n".join(lines)


def get_latest_record_date_iso(project: str) -> str:
    """ISO-8601 timestamp of this project's most recent confirm — for
    response_builder's meta.dataRefreshedAt. Falls back to now if the
    project has no recorded refresh time yet."""
    context = load_data_context(project)
    return context.get("last_refreshed") or datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


def data_freshness_note(project: str) -> str:
    """Compact 'today vs latest available data' line for prompts that write
    user-facing narrative but don't otherwise see the full data context
    (answer_enrichment_task, out_of_scope_response_task)."""
    now = datetime.now()
    latest_iso = get_latest_record_date_iso(project)
    try:
        latest = datetime.strptime(latest_iso, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        latest = now
    return (
        f"Today's date is {now.strftime('%d %B %Y')}. "
        f"The latest data available for this project was loaded on {latest.strftime('%d %B %Y')}."
    )
