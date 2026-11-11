"""Two-step query split: SQL generation (DS) → execution (BE) → enrichment (DS)."""

from typing import Any, Optional

from pydantic import BaseModel

from src.schemas.user_request import QueryResponse


class SqlStageResponse(BaseModel):
    """Returned by POST /query/{project}/sql.

    is_complete=True  → response is already the final answer (out-of-scope,
                        external source, RAG, or session-expired). BE returns
                        it directly to the frontend.
    is_complete=False → sql holds the demasked SELECT for BE to execute.
                        BE sends rows back to POST /query/{project}/enrich.
    """
    answer_id: str
    question_id: str
    is_complete: bool
    max_context_window: bool = False
    chat_expire: bool = False
    response: Optional[QueryResponse] = None
    sql: Optional[str] = None


class EnrichRequest(BaseModel):
    """Sent by BE to POST /query/{project}/enrich after executing the SQL."""
    answer_id: str
    question_id: str
    user_id: str
    session_id: str
    mode: str
    query: str
    sql: str
    rows: list[dict[str, Any]]
    row_count: int
