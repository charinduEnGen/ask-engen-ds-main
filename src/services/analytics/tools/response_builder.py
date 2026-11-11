"""Builds the final response envelope from a result dataframe.

Moved from src/tools/response_builder.py; that file is now a backward-compat
re-export.
"""

import json
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.schemas.user_request import (
    QueryResponse,
    ResponseMeta,
)
from src.utils import kb_loader

_CURRENCY_COLUMNS = {
    "revenue",
    "amount",
    "transaction_value",
    "cost",
    "balance",
    "price",
    "profit",
    "fee",
    "rental",
    "projected_monthly_revenue",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _humanize_column(col: str) -> str:
    base, lower = col, col.lower()
    currency_suffix = f"({settings.default_currency.lower()})"
    has_curr_suffix = lower.endswith(currency_suffix) or lower.endswith(f"_{settings.default_currency.lower()}")
    if lower.endswith(currency_suffix):
        base = col[: -len(currency_suffix)]
    elif lower.endswith(f"_{settings.default_currency.lower()}"):
        base = col[: -len(f"_{settings.default_currency.lower()}")]

    label = base.replace("_", " ").strip().title()
    if has_curr_suffix or lower in _CURRENCY_COLUMNS:
        label = f"{label} ({settings.default_currency})"
    return label


def _to_number(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _format_label(value) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%b %Y").upper()
    return str(value)


def dataframe_to_table_columns(df: pd.DataFrame) -> list[dict]:
    return [{"key": col, "label": _humanize_column(col)} for col in df.columns]


def dataframe_to_rows(df: pd.DataFrame) -> list[dict]:
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
    return json.loads(json.dumps(records, default=str))


def build_table_payload(
    result_df: pd.DataFrame,
    page: int = 1,
    page_size: int = settings.default_page_size,
    paginated: bool = False,
    title: Optional[str] = None,
) -> dict:
    total_rows = len(result_df)
    table: dict = {"title": title}

    if paginated:
        table["columns"] = dataframe_to_table_columns(result_df)
        page_df = result_df.iloc[(page - 1) * page_size : page * page_size]
        table["rows"] = dataframe_to_rows(page_df)
        total_pages = max(1, -(-total_rows // page_size))
        table["pagination"] = {
            "page": page,
            "pageSize": page_size,
            "totalRows": total_rows,
            "totalPages": total_pages,
            "hasNext": page < total_pages,
            "hasPrevious": page > 1,
        }
    else:
        table["columns"] = dataframe_to_table_columns(result_df)
        table["rows"] = dataframe_to_rows(result_df)
        table["page"] = page
        table["pageSize"] = page_size
        table["totalRows"] = total_rows

    return table


def build_chart_payload(
    chart_type: str,
    title: str,
    result_df: pd.DataFrame,
    label_column: str,
    value_column: str,
    unit: str,
) -> dict:
    if chart_type == "pie_chart":
        sorted_df = result_df.sort_values(value_column, ascending=False)
        segments = [
            {"label": _format_label(row[label_column]), "value": _to_number(row[value_column])}
            for _, row in sorted_df.iterrows()
        ]
        return {
            "type": "pie_chart",
            "title": title,
            "pie": {"data": {"unit": unit, "segments": segments}},
            "bar": None,
            "line": None,
        }

    if chart_type == "bar_chart":
        sorted_df = result_df.sort_values(value_column, ascending=False)
        bars = [
            {"label": _format_label(row[label_column]), "value": _to_number(row[value_column])}
            for _, row in sorted_df.iterrows()
        ]
        return {
            "type": "bar_chart",
            "title": title,
            "unit": unit,
            "pie": None,
            "bar": bars,
            "line": None,
        }

    sorted_df = result_df.sort_values(label_column, ascending=True)
    points = [
        {"label": _format_label(row[label_column]), "value": _to_number(row[value_column])}
        for _, row in sorted_df.iterrows()
    ]
    return {
        "type": "line_chart",
        "title": title,
        "unit": unit,
        "pie": None,
        "bar": None,
        "line": {"data": {"unit": unit, "points": points}},
    }


def build_response(
    *,
    answer_id: str,
    question_id: str,
    user_id: str,
    session_id: str,
    mode: str,
    max_context_window: bool,
    chat_expire: bool,
    text_content: Optional[str],
    table: Optional[dict],
    chart: Optional[dict],
    follow_ups: Optional[list[str]],
    merchant_disambiguation: Optional[dict] = None,
) -> QueryResponse:
    return QueryResponse(
        answerId=answer_id,
        questionId=question_id,
        userId=user_id,
        sessionId=session_id,
        meta=ResponseMeta(
            mode=mode,
            dataRefreshedAt=kb_loader.get_latest_record_date_iso(mode),
            generatedAt=_now_iso(),
        ),
        maxContextWindow=max_context_window,
        chatExpire=chat_expire,
        merchantDisambiguation=merchant_disambiguation,
        answer={"text": {"content": text_content}, "table": table, "chart": chart},
        followUps=follow_ups,
    )
