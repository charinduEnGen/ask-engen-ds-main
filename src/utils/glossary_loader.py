"""Loads and saves the project-level business glossary from/to
data/<project>/business_glossary.json.

The glossary is a list of term entries; each entry has a `term` (the
business label), an optional `definition`, and an optional `notes` field for
internal caveats. When injected into the LLM prompt it is rendered as a
compact block so the model understands domain-specific terminology and logic
before writing SQL.
"""

import json
from pathlib import Path

from src.config import DATA_DIR


def _glossary_path(project: str) -> Path:
    return DATA_DIR / project / "business_glossary.json"


def load_glossary(project: str) -> list[dict]:
    path = _glossary_path(project)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_glossary(project: str, entries: list[dict]) -> None:
    path = _glossary_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def render_glossary_block(project: str) -> str:
    """Returns a compact text block for injection into the LLM's static
    context. Returns an empty string if no glossary exists yet."""
    entries = load_glossary(project)
    if not entries:
        return ""
    lines = ["Business glossary / internal terminology:"]
    for entry in entries:
        term = entry.get("term", "").strip()
        if not term:
            continue
        definition = entry.get("definition", "").strip()
        notes = entry.get("notes", "").strip()
        line = f"- {term}"
        if definition:
            line += f": {definition}"
        if notes:
            line += f" [{notes}]"
        lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""
