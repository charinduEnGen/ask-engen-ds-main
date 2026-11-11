"""FastAPI routes for the Knowledge (Vector DB) service.

POST /knowledge/ingest           — upload + embed document(s)
GET  /knowledge/{project}        — list indexed docs
DELETE /knowledge/{project}/{id} — remove a doc
"""

from __future__ import annotations

import io
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.db import vector_store
from src.utils.logging_utils import console_log

router = APIRouter(prefix="/knowledge", tags=["Knowledge"])

_SUPPORTED: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".pdf": "pdf",
    ".docx": "docx",
}


def _extract_text(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    kind = _SUPPORTED.get(ext)
    if kind is None:
        supported = ", ".join(_SUPPORTED)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {supported}",
        )

    if kind == "text":
        return content.decode("utf-8", errors="replace")

    if kind == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="pypdf is not installed — run: pip install pypdf",
            ) from exc
        reader = PdfReader(io.BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if kind == "docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise HTTPException(
                status_code=500,
                detail="python-docx is not installed — run: pip install python-docx",
            ) from exc
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise HTTPException(status_code=400, detail=f"Unhandled file type: {ext}")


@router.post("/ingest")
async def ingest_documents(
    project_name: str = Form(...),
    files: list[UploadFile] = File(...),
) -> dict:
    project = project_name.strip()
    if not project:
        raise HTTPException(status_code=400, detail="project_name is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    ingested = []
    for upload in files:
        filename = upload.filename or "document.txt"
        content = await upload.read()
        try:
            text = _extract_text(filename, content)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Could not read '{filename}': {exc}"
            ) from exc

        if not text.strip():
            raise HTTPException(
                status_code=400, detail=f"'{filename}' produced no readable text"
            )

        doc_id = uuid.uuid4().hex
        entry = vector_store.add_document(project, doc_id, filename, text)
        console_log(
            project, "KNOWLEDGE_INGEST",
            f"doc_id={doc_id} filename={filename} chunks={entry['chunk_count']}",
        )
        ingested.append(entry)

    return {"success": True, "project_name": project, "documents": ingested}


@router.get("/{project}")
async def list_documents(project: str) -> dict:
    docs = vector_store.list_documents(project)
    return {"project": project, "documents": docs}


@router.delete("/{project}/{doc_id}")
async def delete_document(project: str, doc_id: str) -> dict:
    removed = vector_store.delete_document(project, doc_id)
    return {"success": True, "removed": removed}
