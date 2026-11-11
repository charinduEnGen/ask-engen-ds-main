"""Dataset ingestion and confirmation for the analytics service.

Moved from src/routes/datasets.py; that file is now a backward-compat
re-export.
"""

import asyncio
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import DATA_DIR, settings
from src.db.engine import get_engine, reset_engine
from src.core.llm.gate import call_llm_guarded
from src.utils import prompt_loader
from src.utils.external_db_loader import (
    load_external_db,
    save_external_db,
)
from src.core.llm.client import LLMProviderError, TokenLimitExceededError, call_llm
from src.utils.logging_utils import console_log

router = APIRouter(prefix="/datasets", tags=["Datasets"])

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_CSV_EXTS = {".csv"}
_EXCEL_EXTS = {".xlsx", ".xls"}

# All analytics data lives under this fixed namespace so the query
# agent (which routes by mode="analytics") always finds it.
_NS = "analytics"


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug or "table"


def _table_name(filename: str, sheet_name: str | None) -> str:
    stem = Path(filename).stem
    if sheet_name is None:
        return _slugify(stem)
    return f"{_slugify(stem)}_{_slugify(sheet_name)}"


def _read_sheets(path: Path, nrows: int | None = None) -> dict[str | None, pd.DataFrame]:
    ext = path.suffix.lower()
    if ext in _CSV_EXTS:
        return {None: pd.read_csv(path, nrows=nrows)}
    if ext in _EXCEL_EXTS:
        return pd.read_excel(path, sheet_name=None, nrows=nrows)
    raise ValueError(f"Unsupported file type '{ext}' — only .xlsx, .xls, and .csv are supported")


def _staging_dir(upload_id: str) -> Path:
    return DATA_DIR / _NS / "_staging" / upload_id


def _table_dir(table: str) -> Path:
    return DATA_DIR / _NS / table


def _dtype_label(series: pd.Series) -> str:
    kind = series.dtype.kind
    if kind in "iu":
        return "integer"
    if kind == "f":
        return "decimal"
    if kind == "b":
        return "boolean"
    if kind == "M":
        return "datetime"
    return "text"


def _sample_values(series: pd.Series, limit: int = 6) -> list[str]:
    values = [str(v) for v in series.dropna().unique()[:limit]]
    return values


class IngestResponse(BaseModel):
    project_name: str
    upload_id: str
    tables: list[dict[str, Any]]
    relationships: list[dict[str, Any]]
    data_context: dict[str, Any]
    patterns: list[dict[str, Any]]


@router.post("/ingest", response_model=IngestResponse)
async def ingest(project_name: str = Form(...), files: list[UploadFile] = File(...)) -> IngestResponse:
    project = project_name.strip()
    if not project:
        raise HTTPException(status_code=400, detail="project_name is required")
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")

    upload_id = uuid.uuid4().hex
    staging_dir = _staging_dir(upload_id)
    staging_dir.mkdir(parents=True, exist_ok=True)

    tables_preview: list[dict[str, Any]] = []
    try:
        for upload in files:
            filename = upload.filename or "upload.xlsx"
            dest = staging_dir / filename
            content = await upload.read()
            dest.write_bytes(content)

            try:
                sheets = _read_sheets(dest, nrows=settings.schema_sample_rows)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"Could not read '{filename}': {exc}") from exc

            for sheet_name, head_df in sheets.items():
                table = _table_name(filename, sheet_name)
                columns = [
                    {
                        "name": col,
                        "dtype": _dtype_label(head_df[col]),
                        "samples": _sample_values(head_df[col]),
                    }
                    for col in head_df.columns
                ]
                tables_preview.append(
                    {
                        "table": table,
                        "source_file": filename,
                        "sheet": str(sheet_name) if sheet_name is not None else "",
                        "columns": columns,
                    }
                )
    except HTTPException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    if not tables_preview:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="No sheets found in the uploaded file(s)")

    discovered = await _run_schema_discovery(_NS, tables_preview)

    return IngestResponse(
        project_name=project,
        upload_id=upload_id,
        tables=discovered.get("tables", []),
        relationships=discovered.get("relationships", []),
        data_context=discovered.get("data_context", {}),
        patterns=discovered.get("patterns", []),
    )


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


async def _run_schema_discovery(project: str, tables_preview: list[dict[str, Any]]) -> dict:
    system_prompt = prompt_loader.build_agent_system_prompt("schema_relation_builder_agent")
    prompt = prompt_loader.render_task_prompt(
        "schema_discovery_task",
        project_name=project,
        sample_row_count=str(settings.schema_sample_rows),
        tables_preview=json.dumps(tables_preview, ensure_ascii=False),
    )

    try:
        result = await call_llm_guarded(
            call_llm,
            project,
            system_prompt,
            prompt,
            gemini_model=settings.gemini_model_data_processing,
            call_timeout=settings.schema_discovery_timeout_seconds,
        )
    except (TokenLimitExceededError, LLMProviderError) as exc:
        raise HTTPException(status_code=502, detail=f"Schema discovery failed: {exc}") from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"Schema discovery timed out after {settings.schema_discovery_timeout_seconds:.0f}s — "
                "try again, or upload fewer files/sheets at once."
            ),
        ) from exc

    console_log(project, "SCHEMA_DISCOVERY", f"raw decision:\n{result.text}")

    def _parse_json(text: str) -> dict:
        raw = _strip_fences(text)
        try:
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            pass
        try:
            from json_repair import repair_json  # type: ignore
            repaired = repair_json(raw, return_objects=True)
            if isinstance(repaired, dict):
                console_log(project, "SCHEMA_DISCOVERY", "JSON repaired by json_repair")
                return repaired
        except Exception:
            pass
        return None

    decision = _parse_json(result.text)

    if decision is None:
        console_log(project, "SCHEMA_DISCOVERY", "JSON parse failed, retrying LLM call")
        try:
            result = await call_llm_guarded(
                call_llm,
                project,
                system_prompt,
                prompt,
                gemini_model=settings.gemini_model_data_processing,
                call_timeout=settings.schema_discovery_timeout_seconds,
            )
        except (TokenLimitExceededError, LLMProviderError) as exc:
            raise HTTPException(status_code=502, detail=f"Schema discovery retry failed: {exc}") from exc
        decision = _parse_json(result.text)

    if decision is None:
        raise HTTPException(
            status_code=502,
            detail="Schema discovery returned invalid JSON after repair and retry. Try uploading fewer sheets at once.",
        )

    return decision


class ConfirmRequest(BaseModel):
    project_name: str
    upload_id: str
    tables: list[dict[str, Any]]
    relationships: list[dict[str, Any]] = []
    data_context: dict[str, Any] = {}
    patterns: list[dict[str, Any]] = []


class ConfirmResponse(BaseModel):
    success: bool
    project_name: str
    tables: list[str]


@router.post("/confirm", response_model=ConfirmResponse)
async def confirm(body: ConfirmRequest) -> ConfirmResponse:
    if not body.project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required")

    staging_dir = _staging_dir(body.upload_id)
    if not staging_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="Upload not found or already confirmed — re-run /datasets/ingest first",
        )

    staged_files = list(staging_dir.iterdir())
    if not staged_files:
        raise HTTPException(status_code=400, detail="Staged upload directory is empty")

    reset_engine(_NS)
    engine = get_engine(_NS)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    saved_tables: list[str] = []
    table_facts: dict[str, dict[str, Any]] = {}

    for table_cfg in body.tables:
        table_name = table_cfg.get("name")
        if not table_name:
            continue

        source_file, sheet_name = _find_source(table_name, staged_files)
        if source_file is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not match table '{table_name}' back to an uploaded file/sheet",
            )

        full_df = (
            pd.read_csv(source_file) if sheet_name is None else pd.read_excel(source_file, sheet_name=sheet_name)
        )
        full_df.to_sql(table_name, engine, if_exists="replace", index=False)
        _create_indexes_heuristic(engine, table_name, list(full_df.columns))

        table_dir = _table_dir(table_name)
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "schema.json").write_text(
            json.dumps(
                {"type": table_cfg.get("type"), "columns": table_cfg.get("columns", [])}, indent=2
            ),
            encoding="utf-8",
        )

        table_facts[table_name] = {
            "row_count": len(full_df),
            "source_file": source_file.name,
            "sheet": sheet_name,
        }
        saved_tables.append(table_name)

    ns_dir = DATA_DIR / _NS
    ns_dir.mkdir(parents=True, exist_ok=True)
    (ns_dir / "data_context.json").write_text(
        json.dumps(
            {
                "description": body.data_context.get("description", ""),
                "currency": body.data_context.get("currency"),
                "time_grain": body.data_context.get("time_grain", "unknown"),
                "notes": body.data_context.get("notes", []),
                "tables": table_facts,
                "last_refreshed": now_iso,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ns_dir / "patterns.json").write_text(json.dumps({"patterns": body.patterns}, indent=2), encoding="utf-8")
    (ns_dir / "relationships.json").write_text(json.dumps({"relationships": body.relationships}, indent=2), encoding="utf-8")

    shutil.rmtree(staging_dir, ignore_errors=True)

    return ConfirmResponse(success=True, project_name=_NS, tables=saved_tables)


class ExternalSourceRequest(BaseModel):
    project_name: str
    key_name: str
    description: str = ""
    link: str = ""


class ExternalSourceResponse(BaseModel):
    success: bool
    project_name: str
    key_name: str
    files: list[dict[str, Any]]


@router.post("/external", response_model=ExternalSourceResponse)
async def register_external_source(body: ExternalSourceRequest) -> ExternalSourceResponse:
    project = body.project_name.strip()
    key_name = body.key_name.strip()
    if not project or not key_name:
        raise HTTPException(status_code=400, detail="project_name and key_name are required")
    if not body.link.strip():
        raise HTTPException(status_code=400, detail="link (URL) is required")

    sources = load_external_db(project)
    replaced = False
    entry = {
        "key_name": key_name,
        "description": body.description,
        "link": body.link.strip(),
    }
    for i, s in enumerate(sources):
        if s.get("key_name") == key_name:
            sources[i] = entry
            replaced = True
            break
    if not replaced:
        sources.append(entry)
    save_external_db(project, sources)

    console_log(project, "EXTERNAL_SOURCE", f"registered key={key_name} link={body.link} replaced={replaced}")
    return ExternalSourceResponse(success=True, project_name=project, key_name=key_name, files=[])


@router.get("/external/{project}")
async def list_external_sources(project: str) -> dict:
    return {"project": project, "sources": load_external_db(project)}


@router.delete("/external/{project}/{key_name}")
async def delete_external_source(project: str, key_name: str) -> dict:
    sources = load_external_db(project)
    before = len(sources)
    sources = [s for s in sources if s.get("key_name") != key_name]
    save_external_db(project, sources)
    return {"success": True, "removed": before - len(sources)}


_INDEX_COL_PATTERNS = (
    "id", "_id", "key", "_key", "code", "_code",
    "date", "_date", "time", "_time", "at", "_at",
    "ref", "_ref",
)


def _create_indexes_heuristic(engine, table_name: str, columns: list[str]) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for col in columns:
            col_lower = col.lower()
            if not any(col_lower == p or col_lower.endswith(p) for p in _INDEX_COL_PATTERNS):
                continue
            idx_name = f"idx_{table_name}_{col_lower}"
            conn.execute(
                text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col}")')
            )


def _find_source(table_name: str, staged_files: list[Path]) -> tuple[Path | None, str | None]:
    for path in staged_files:
        if path.suffix.lower() in _CSV_EXTS:
            if _table_name(path.name, None) == table_name:
                return path, None
            continue
        try:
            sheet_names = pd.ExcelFile(path).sheet_names
        except Exception:  # noqa: BLE001
            continue
        for sheet_name in sheet_names:
            if _table_name(path.name, sheet_name) == table_name:
                return path, sheet_name
    return None, None
