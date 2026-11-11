"""CRUD endpoints for the project-level business glossary.

GET  /glossary/{project}        — returns current glossary entries
PUT  /glossary/{project}        — replaces entire glossary with the new list
POST /glossary/{project}/entry  — appends a single entry
DELETE /glossary/{project}/{term} — removes the entry whose term matches
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.utils.glossary_loader import load_glossary, save_glossary

router = APIRouter(prefix="/glossary", tags=["Glossary"])


class GlossaryEntry(BaseModel):
    term: str
    definition: str = ""
    notes: str = ""


class GlossaryList(BaseModel):
    entries: list[GlossaryEntry]


@router.get("/{project}")
async def get_glossary(project: str) -> dict:
    return {"project": project, "entries": load_glossary(project)}


@router.put("/{project}")
async def replace_glossary(project: str, body: GlossaryList) -> dict:
    entries = [e.model_dump() for e in body.entries]
    save_glossary(project, entries)
    return {"success": True, "project": project, "count": len(entries)}


@router.post("/{project}/entry")
async def add_entry(project: str, entry: GlossaryEntry) -> dict:
    entries = load_glossary(project)
    term_lower = entry.term.strip().lower()
    if not term_lower:
        raise HTTPException(status_code=400, detail="term is required")
    # Replace if term already exists (case-insensitive), else append
    replaced = False
    for i, e in enumerate(entries):
        if e.get("term", "").lower() == term_lower:
            entries[i] = entry.model_dump()
            replaced = True
            break
    if not replaced:
        entries.append(entry.model_dump())
    save_glossary(project, entries)
    return {"success": True, "replaced": replaced, "count": len(entries)}


@router.delete("/{project}/{term}")
async def delete_entry(project: str, term: str) -> dict:
    entries = load_glossary(project)
    term_lower = term.strip().lower()
    before = len(entries)
    entries = [e for e in entries if e.get("term", "").lower() != term_lower]
    save_glossary(project, entries)
    return {"success": True, "removed": before - len(entries)}
