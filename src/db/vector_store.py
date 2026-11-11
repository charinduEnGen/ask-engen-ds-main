"""File-based vector store for the Knowledge (RAG) service.

Layout per project:
  data/<project>/knowledge/
    index.json            — [{id, name, created_at, chunk_count, status}]
    <doc_id>/
      meta.json           — {id, name, created_at}
      chunks.json         — [{text, embedding: [float...]}]

Embeddings use Gemini text-embedding-004. On a non-Gemini provider or when
the key is absent, a zero-vector fallback is used so the rest of the pipeline
still runs (search results will be meaningless, but ingest won't crash).
"""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DATA_DIR, settings

_EMBED_MODEL = "models/text-embedding-004"
_CHUNK_SIZE = 800    # characters per chunk
_CHUNK_OVERLAP = 100  # overlap between adjacent chunks


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _knowledge_root(project: str) -> Path:
    return DATA_DIR / project / "knowledge"


def _doc_dir(project: str, doc_id: str) -> Path:
    return _knowledge_root(project) / doc_id


def _index_path(project: str) -> Path:
    return _knowledge_root(project) / "index.json"


# ---------------------------------------------------------------------------
# Index (metadata list)
# ---------------------------------------------------------------------------

def _load_index(project: str) -> list[dict]:
    p = _index_path(project)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save_index(project: str, index: list[dict]) -> None:
    p = _index_path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Uses Gemini; falls back to zero vectors."""
    if not texts:
        return []
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            BATCH = 100
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), BATCH):
                result = genai.embed_content(
                    model=_EMBED_MODEL,
                    content=texts[i : i + BATCH],
                    task_type="retrieval_document",
                )
                batch_embs = result.get("embedding") or []
                # embed_content may return a single list when given one text
                if batch_embs and not isinstance(batch_embs[0], list):
                    batch_embs = [batch_embs]
                all_embeddings.extend(batch_embs)
            return all_embeddings
        except Exception:  # noqa: BLE001
            pass
    # Zero-vector fallback
    return [[0.0] * 768 for _ in texts]


def _embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.gemini_api_key)
            result = genai.embed_content(
                model=_EMBED_MODEL,
                content=text,
                task_type="retrieval_query",
            )
            emb = result.get("embedding") or []
            if emb and isinstance(emb[0], list):
                emb = emb[0]
            return emb
        except Exception:  # noqa: BLE001
            pass
    return [0.0] * 768


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_document(project: str, doc_id: str, filename: str, text: str) -> dict[str, Any]:
    """Chunk, embed, and persist a document. Returns the index entry."""
    chunks = _chunk_text(text)
    if not chunks:
        raise ValueError("Document produced no text chunks after splitting")

    embeddings = _embed_batch(chunks)
    chunk_records = [{"text": t, "embedding": e} for t, e in zip(chunks, embeddings)]

    doc_path = _doc_dir(project, doc_id)
    doc_path.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()
    meta: dict[str, Any] = {
        "id": doc_id,
        "name": filename,
        "created_at": now_iso,
        "chunk_count": len(chunks),
    }
    (doc_path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (doc_path / "chunks.json").write_text(
        json.dumps(chunk_records, ensure_ascii=False), encoding="utf-8"
    )

    entry = {**meta, "status": "indexed"}
    index = _load_index(project)
    index = [e for e in index if e.get("id") != doc_id]
    index.append(entry)
    _save_index(project, index)
    return entry


def list_documents(project: str) -> list[dict]:
    return _load_index(project)


def delete_document(project: str, doc_id: str) -> bool:
    doc_path = _doc_dir(project, doc_id)
    if doc_path.exists():
        shutil.rmtree(doc_path)
    index = _load_index(project)
    before = len(index)
    index = [e for e in index if e.get("id") != doc_id]
    _save_index(project, index)
    return len(index) < before


def search(project: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k most relevant chunks across all docs for this project."""
    index = _load_index(project)
    if not index:
        return []

    query_emb = _embed_query(query)
    results: list[dict[str, Any]] = []

    for entry in index:
        doc_id = entry["id"]
        chunks_path = _doc_dir(project, doc_id) / "chunks.json"
        if not chunks_path.exists():
            continue
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        for chunk in chunks:
            sim = _cosine(query_emb, chunk.get("embedding", []))
            results.append({
                "doc_id": doc_id,
                "doc_name": entry["name"],
                "text": chunk["text"],
                "score": sim,
            })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]
