"""Live database connection route — Oracle and MySQL only.

Moved from src/routes/db_connect.py; that file is now a backward-compat
re-export.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config import DATA_DIR, settings
from src.db.engine import get_engine, reset_engine
from src.services.analytics.routes.datasets import (
    _NS,
    _dtype_label,
    _run_schema_discovery,
    _sample_values,
)

router = APIRouter(prefix="/db-connect", tags=["DBConnect"])


class DBConfig(BaseModel):
    db_type: str
    host: str
    port: int
    database: str
    schema: str | None = None
    username: str
    password: str
    project_name: str


class IngestRequest(DBConfig):
    tables: list[str]


def _connect(cfg: DBConfig):
    if cfg.db_type == "mysql":
        try:
            import mysql.connector  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="mysql-connector-python is not installed on the server.",
            ) from exc
        try:
            return mysql.connector.connect(
                host=cfg.host,
                port=cfg.port,
                database=cfg.database,
                user=cfg.username,
                password=cfg.password,
                connect_timeout=10,
                connection_timeout=10,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"MySQL connection failed: {exc}") from exc

    if cfg.db_type == "oracle":
        try:
            import oracledb  # type: ignore
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail="oracledb is not installed on the server.",
            ) from exc
        try:
            dsn = f"{cfg.host}:{cfg.port}/{cfg.database}"
            return oracledb.connect(user=cfg.username, password=cfg.password, dsn=dsn)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Oracle connection failed: {exc}") from exc

    raise HTTPException(status_code=400, detail=f"Unsupported db_type: {cfg.db_type!r}. Use 'mysql' or 'oracle'.")


def _list_tables(cursor, db_type: str, schema: str | None = None) -> list[str]:
    if db_type == "mysql":
        cursor.execute("SHOW TABLES")
        return [row[0] for row in cursor.fetchall()]
    if schema:
        cursor.execute(
            "SELECT table_name FROM all_tables WHERE owner = :owner ORDER BY table_name",
            {"owner": schema.upper()},
        )
    else:
        cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
    return [row[0] for row in cursor.fetchall()]


def _sample_table(cursor, table_name: str, db_type: str, schema: str | None = None) -> dict:
    try:
        if db_type == "mysql":
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 5")
        else:
            qualified = f'"{schema.upper()}"."{table_name}"' if schema else f'"{table_name}"'
            cursor.execute(f"SELECT * FROM {qualified} FETCH FIRST 5 ROWS ONLY")
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()

        def _cell(v):
            if v is None:
                return None
            if hasattr(v, "read"):
                return v.read()
            return str(v)

        sample = [
            {c: _cell(v) for c, v in zip(cols, row)}
            for row in rows
        ]
        return {"name": table_name, "columns": cols, "sample_rows": sample}
    except Exception as exc:
        return {"name": table_name, "columns": [], "sample_rows": [], "error": str(exc)}


@router.post("/schema")
async def get_schema(cfg: DBConfig) -> dict:
    conn = _connect(cfg)
    cursor = conn.cursor()
    table_names = _list_tables(cursor, cfg.db_type, cfg.schema)
    tables = [_sample_table(cursor, t, cfg.db_type, cfg.schema) for t in table_names]
    cursor.close()
    conn.close()
    return {"status": "connected", "db_type": cfg.db_type, "database": cfg.database, "tables": tables}


class SaveSchemaRequest(BaseModel):
    project_name: str
    tables: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    data_context: dict[str, Any] = {}
    patterns: list[dict[str, Any]] = []


@router.post("/save-schema")
async def save_schema_metadata(body: SaveSchemaRequest) -> dict:
    if not body.project_name.strip():
        raise HTTPException(status_code=400, detail="project_name is required")

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    saved_tables: list[str] = []

    for table_cfg in body.tables:
        table_name = table_cfg.get("name")
        if not table_name:
            continue
        table_dir: Path = DATA_DIR / _NS / table_name
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "schema.json").write_text(
            json.dumps(
                {"type": table_cfg.get("type", "fact"), "columns": table_cfg.get("columns", [])},
                indent=2,
            ),
            encoding="utf-8",
        )
        saved_tables.append(table_name)

    project_dir: Path = DATA_DIR / _NS
    project_dir.mkdir(parents=True, exist_ok=True)

    (project_dir / "data_context.json").write_text(
        json.dumps(
            {
                "description": body.data_context.get("description", ""),
                "currency": body.data_context.get("currency"),
                "time_grain": body.data_context.get("time_grain", "unknown"),
                "notes": body.data_context.get("notes", []),
                "tables": {t: {} for t in saved_tables},
                "last_refreshed": now_iso,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (project_dir / "patterns.json").write_text(
        json.dumps({"patterns": body.patterns}, indent=2), encoding="utf-8"
    )
    (project_dir / "relationships.json").write_text(
        json.dumps({"relationships": body.relationships}, indent=2), encoding="utf-8"
    )

    return {"success": True, "project_name": _NS, "tables": saved_tables}


_INDEX_COL_PATTERNS = (
    "id", "_id", "key", "_key", "code", "_code",
    "date", "_date", "time", "_time", "at", "_at",
    "ref", "_ref",
)


def _create_indexes(engine, table_name: str, columns: list[str], authoritative: bool = False) -> None:
    from sqlalchemy import text

    with engine.begin() as conn:
        for col in columns:
            col_lower = col.lower()
            if not authoritative:
                if not any(col_lower == p or col_lower.endswith(p) for p in _INDEX_COL_PATTERNS):
                    continue
            idx_name = f"idx_{table_name}_{col_lower}"
            conn.execute(
                text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ("{col}")')
            )


@router.post("/ingest")
async def ingest_from_db(req: IngestRequest) -> dict:
    try:
        import pandas as pd
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="pandas is not installed.") from exc

    conn = _connect(req)
    reset_engine(_NS)
    engine = get_engine(_NS)

    tables_preview: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    CHUNK = 5_000

    def _coerce_lobs(chunk: "pd.DataFrame") -> "pd.DataFrame":
        for col in chunk.columns:
            if chunk[col].dtype == object:
                chunk[col] = chunk[col].apply(
                    lambda v: v.read() if hasattr(v, "read") else v
                )
        return chunk

    for table_name in req.tables:
        try:
            if req.db_type == "mysql":
                sql = f"SELECT * FROM `{table_name}`"
            else:
                qualified = f'"{req.schema.upper()}"."{table_name}"' if req.schema else f'"{table_name}"'
                sql = f"SELECT * FROM {qualified}"

            ingested_cols: list[str] = []
            first = True
            for chunk in pd.read_sql(sql, conn, chunksize=CHUNK):
                if req.db_type == "oracle":
                    chunk = _coerce_lobs(chunk)
                chunk.to_sql(
                    table_name.lower(), engine,
                    if_exists="replace" if first else "append",
                    index=False,
                    method="multi",
                )
                if first:
                    ingested_cols = list(chunk.columns)
                    first = False

            if not ingested_cols:
                raise ValueError("Table returned no rows")

            _create_indexes(engine, table_name.lower(), ingested_cols)

            n = settings.schema_sample_rows
            with engine.connect() as lite_conn:
                sample_df = pd.read_sql(
                    f'SELECT * FROM "{table_name.lower()}" LIMIT {n}',
                    lite_conn,
                )

            tables_preview.append({
                "table": table_name,
                "columns": [
                    {
                        "name": col,
                        "dtype": _dtype_label(sample_df[col]),
                        "samples": _sample_values(sample_df[col]),
                    }
                    for col in sample_df.columns
                ],
            })
        except Exception as exc:
            errors.append({"table": table_name, "error": str(exc)})

    conn.close()

    if not tables_preview:
        raise HTTPException(status_code=400, detail=f"No tables could be ingested. Errors: {errors}")

    discovered = await _run_schema_discovery(_NS, tables_preview)

    for tbl in discovered.get("tables", []):
        tbl_name = tbl.get("table", "").lower()
        if not tbl_name:
            continue
        pk_fk_time_cols = [
            col["name"]
            for col in tbl.get("columns", [])
            if col.get("key_type", "").upper() in {"PK", "FK", "TIME"}
        ]
        if pk_fk_time_cols:
            _create_indexes(engine, tbl_name, pk_fk_time_cols, authoritative=True)

    return {
        "status": "ingested",
        "project_name": _NS,
        "upload_id": "__db_connect__",
        "tables": discovered.get("tables", []),
        "relationships": discovered.get("relationships", []),
        "data_context": discovered.get("data_context", {}),
        "patterns": discovered.get("patterns", []),
        "errors": errors,
    }
