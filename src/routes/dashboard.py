"""Analytics dashboard: reads every costs/cost_YYYY-MM-DD.jsonl
(src.utils.cost_utils) and logs/timing_YYYY-MM-DD.jsonl (src.utils.logging_utils)
entry and renders a KPI summary, a cost/time-by-day rollup, and a per-request
Detailed Breakdown table (one row per question/query, with its own
tokens/time/cost) as a single self-contained HTML page. No JS framework, no
template engine — plain server-rendered HTML/CSS so the page has no build
step and no external dependency beyond FastAPI itself.

Also still reads the legacy costs/YYYY-MM-DD.yml and logs/YYYY-MM-DD.yml
files (the format both modules wrote before the JSONL migration) so history
recorded before the switch keeps showing up instead of silently vanishing
from the rollups the day this shipped.

Entries written before `mode` was added to the schema (or from any future
caller that omits it) fall back to the literal "unknown" bucket rather than
being dropped, so historical data still shows up.
"""

from __future__ import annotations

import html
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml
from fastapi import APIRouter, Query
from fastapi.responses import HTMLResponse

from src.config import COSTS_DIR, LOGS_DIR, settings

router = APIRouter(tags=["Dashboard"])

_UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# Data loading / aggregation
# --------------------------------------------------------------------------


def _entry_date(stem: str) -> str:
    """cost_2026-08-14 / timing_2026-08-14 -> 2026-08-14; a bare
    2026-08-14 (legacy .yml filename) passes through unchanged."""
    return stem.split("_", 1)[1] if "_" in stem else stem


def _load_jsonl_entries(directory: Path) -> list[dict]:
    """Every entry from every *.jsonl file under `directory`, each tagged
    with a `date` key parsed from its filename — log_timing/log_cost always
    name the file after the day they were written on, so this is cheaper and
    more robust than re-parsing each entry's `timestamp`."""
    entries: list[dict] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.jsonl")):
        entry_date = _entry_date(path.stem)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["date"] = entry_date
                entries.append(row)
    return entries


def _load_legacy_yml_entries(directory: Path) -> list[dict]:
    """Pre-JSONL-migration entries — every YYYY-MM-DD.yml file under
    `directory`, in the single-document-list shape log_timing/log_cost used
    to write. Kept read-only so days logged before the migration still show
    up; nothing writes this format anymore."""
    entries: list[dict] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.yml")):
        with open(path, "r", encoding="utf-8") as f:
            rows = yaml.safe_load(f) or []
        for row in rows:
            row = dict(row)
            row["date"] = path.stem
            entries.append(row)
    return entries


def _load_entries(directory: Path) -> list[dict]:
    return _load_jsonl_entries(directory) + _load_legacy_yml_entries(directory)


def _matches_filters(
    entry: dict,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    mode: Optional[str],
) -> bool:
    if start_date and entry["date"] < start_date:
        return False
    if end_date and entry["date"] > end_date:
        return False
    if user_id and entry.get("user_id") != user_id:
        return False
    if session_id and entry.get("session_id") != session_id:
        return False
    if mode and (entry.get("mode") or _UNKNOWN).upper() != mode.upper():
        return False
    return True


@dataclass
class GroupTotals:
    """Cost + time rolled up for one (date, user_id, session_id, mode)
    combination — the finest granularity the dashboard shows."""

    date: str
    user_id: str
    session_id: str
    mode: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _aggregate(cost_entries: list[dict], log_entries: list[dict]) -> list[GroupTotals]:
    """Cost entries (one per LLM call, several per answer) and timing entries
    (one per answer) are two independent streams keyed by
    (date, user_id, session_id, mode) here — a request with zero cost
    entries (shouldn't happen, but degrades gracefully) or zero timing
    entries still shows up from whichever stream it did land in."""
    groups: dict[tuple, GroupTotals] = {}

    def _get(key: tuple) -> GroupTotals:
        if key not in groups:
            groups[key] = GroupTotals(date=key[0], user_id=key[1], session_id=key[2], mode=key[3])
        return groups[key]

    for e in cost_entries:
        key = (e["date"], e.get("user_id") or _UNKNOWN, e.get("session_id") or _UNKNOWN, e.get("mode") or _UNKNOWN)
        g = _get(key)
        g.input_tokens += int(e.get("input_tokens") or 0)
        g.output_tokens += int(e.get("output_tokens") or 0)
        g.cached_tokens += int(e.get("cached_tokens") or 0)
        g.cost_usd += float(e.get("estimated_cost_usd") or 0.0)

    for e in log_entries:
        key = (e["date"], e.get("user_id") or _UNKNOWN, e.get("session_id") or _UNKNOWN, e.get("mode") or _UNKNOWN)
        g = _get(key)
        g.requests += 1
        g.time_seconds += float(e.get("total_seconds") or 0.0)

    return sorted(groups.values(), key=lambda g: (g.date, g.user_id, g.session_id, g.mode), reverse=True)


def _rollup(groups: list[GroupTotals], key_func: Callable[[GroupTotals], str]) -> list[dict]:
    """Re-sums the detailed (date, user, session, mode) groups along a single
    dimension (e.g. just `date`) for the per-dimension summary tables —
    cheap because it folds over the already-aggregated groups rather than
    re-scanning raw entries."""
    agg: dict[str, dict] = defaultdict(
        lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "time_seconds": 0.0}
    )
    for g in groups:
        row = agg[key_func(g)]
        row["requests"] += g.requests
        row["input_tokens"] += g.input_tokens
        row["output_tokens"] += g.output_tokens
        row["cost_usd"] += g.cost_usd
        row["time_seconds"] += g.time_seconds
    return [
        {"label": label, **totals}
        for label, totals in sorted(agg.items(), key=lambda kv: kv[0], reverse=True)
    ]


@dataclass
class RequestRow:
    """One row per answer_id — the actual request grain — used by the
    Detailed Breakdown table so each row can carry its own question/query
    text (which only makes sense per-request, not summed across several)."""

    date: str
    timestamp: str
    user_id: str
    session_id: str
    mode: str
    answer_id: str
    query: str
    tokens: int = 0
    cost_usd: float = 0.0
    time_seconds: float = 0.0


def _aggregate_cost_by_answer(cost_entries: list[dict]) -> dict[str, dict]:
    """Sums the (possibly several, one per LLM call) cost entries sharing an
    answer_id down to one token/cost total per answer_id, for joining onto
    log entries in _build_request_rows."""
    agg: dict[str, dict] = defaultdict(lambda: {"tokens": 0, "cost_usd": 0.0})
    for e in cost_entries:
        row = agg[e.get("answer_id") or _UNKNOWN]
        row["tokens"] += int(e.get("input_tokens") or 0) + int(e.get("output_tokens") or 0)
        row["cost_usd"] += float(e.get("estimated_cost_usd") or 0.0)
    return agg


def _build_request_rows(cost_entries: list[dict], log_entries: list[dict]) -> list[RequestRow]:
    """One row per request (log_timing entry, which always carries the
    question text and is written exactly once per answer_id — see
    query_orchestrator.run_query_pipeline), with cost/tokens joined in by
    answer_id. Sorted most-recent-first."""
    cost_by_answer = _aggregate_cost_by_answer(cost_entries)
    rows = []
    for e in log_entries:
        answer_id = e.get("answer_id") or _UNKNOWN
        cost = cost_by_answer.get(answer_id, {"tokens": 0, "cost_usd": 0.0})
        rows.append(
            RequestRow(
                date=e["date"],
                timestamp=e.get("timestamp") or "",
                user_id=e.get("user_id") or _UNKNOWN,
                session_id=e.get("session_id") or _UNKNOWN,
                mode=e.get("mode") or _UNKNOWN,
                answer_id=answer_id,
                query=e.get("query") or "",
                tokens=cost["tokens"],
                cost_usd=cost["cost_usd"],
                time_seconds=float(e.get("total_seconds") or 0.0),
            )
        )
    return sorted(rows, key=lambda r: r.timestamp, reverse=True)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _fmt_cost(usd: float) -> str:
    return f"${usd:,.4f}"


def _fmt_duration(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if hours or minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _esc(value: Optional[str]) -> str:
    return html.escape(value or "")


# --------------------------------------------------------------------------
# Model status (Active / Inactive badges, purely informational)
# --------------------------------------------------------------------------


def _model_active(deprecation_date: Optional[date]) -> bool:
    """Active when there's no known deprecation date, or today's UTC date
    hasn't reached it yet. Purely a dashboard display — nothing in the
    pipeline itself reads these dates to gate a call; see
    GEMINI_MODEL_*_DEPRECATION_DATE in src.config."""
    if deprecation_date is None:
        return True
    return datetime.now(timezone.utc).date() < deprecation_date


def _model_status_items() -> list[tuple[str, str, bool]]:
    """(label, model_name, is_active) for the two Gemini models this
    pipeline calls — see src.orchestrator.steps for where each is used."""
    return [
        (
            "Agent 1 — Data Processing",
            settings.gemini_model_data_processing,
            _model_active(settings.gemini_model_data_processing_deprecation_date),
        ),
        (
            "Agent 2 — Enrichment",
            settings.gemini_model_enrichment,
            _model_active(settings.gemini_model_enrichment_deprecation_date),
        ),
    ]


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

_PAGE_STYLE = """
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #F4F6F9;
    color: #333333;
  }
  .page {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }
  h1.main-heading {
    text-align: center;
    color: #333333;
    margin: 0 0 20px;
  }
  h2.section-heading {
    color: #555555;
    font-size: 18px;
    margin: 0 0 15px;
  }
  .card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
  }
  .section {
    margin-bottom: 20px;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
  }
  .kpi-label {
    color: #555555;
    font-size: 14px;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 28px;
    font-weight: bold;
    color: #0078FF;
  }
  .filters-form {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 15px;
    align-items: end;
  }
  .field label {
    display: block;
    color: #666666;
    font-size: 13px;
    margin-bottom: 6px;
  }
  .field input, .field select {
    width: 100%;
    background: #FFFFFF;
    border: 1px solid #CCCCCC;
    border-radius: 6px;
    padding: 8px;
    font-size: 14px;
    font-family: Arial, sans-serif;
    color: #333333;
  }
  .field-actions {
    display: flex;
    gap: 10px;
  }
  .btn {
    display: inline-block;
    text-align: center;
    text-decoration: none;
    padding: 9px 18px;
    border-radius: 6px;
    font-size: 14px;
    font-family: Arial, sans-serif;
    border: none;
    cursor: pointer;
  }
  .btn-primary {
    background: #0078FF;
    color: #FFFFFF;
  }
  .btn-secondary {
    background: #FFFFFF;
    color: #666666;
    border: 1px solid #CCCCCC;
  }
  .table-wrap {
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
    overflow: hidden;
    overflow-x: auto;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    background: #FFFFFF;
  }
  thead th {
    background: #0078FF;
    color: #FFFFFF;
    text-align: left;
    padding: 12px;
    font-size: 13px;
    white-space: nowrap;
  }
  tbody td {
    padding: 12px;
    border-bottom: 1px solid #EEEEEE;
    font-size: 14px;
    color: #333333;
    white-space: nowrap;
  }
  tbody tr:last-child td {
    border-bottom: none;
  }
  tbody tr:hover {
    background: #F8F9FA;
  }
  .muted {
    color: #888888;
    font-size: 13px;
  }
  .empty-state {
    color: #888888;
    padding: 20px;
    text-align: center;
  }
  .mode-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    background: #F4F6F9;
    color: #0078FF;
  }
  .query-cell {
    white-space: normal;
    max-width: 420px;
    color: #333333;
  }
  .model-status-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
    padding: 12px 20px;
    margin-bottom: 20px;
  }
  .model-status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #555555;
  }
  .status-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: bold;
  }
  .status-badge.active {
    background: #E6F6EC;
    color: #1A7F37;
  }
  .status-badge.inactive {
    background: #FDECEC;
    color: #C0392B;
  }
"""


def _kpi_card(label: str, value: str) -> str:
    return f"""
    <div class="card">
      <div class="kpi-label">{html.escape(label)}</div>
      <div class="kpi-value">{html.escape(value)}</div>
    </div>"""


def _render_model_status() -> str:
    items = []
    for label, model_name, active in _model_status_items():
        status_class = "active" if active else "inactive"
        status_text = "Active" if active else "Inactive"
        items.append(
            '<div class="model-status-item">'
            f"<span>{_esc(label)} ({_esc(model_name)})</span>"
            f'<span class="status-badge {status_class}">{status_text}</span>'
            "</div>"
        )
    return f"""
    <div class="card model-status-bar">{''.join(items)}</div>"""


def _select_options(values: list[str], selected: Optional[str]) -> str:
    opts = ['<option value="">All</option>']
    for v in values:
        sel = " selected" if selected and v.upper() == selected.upper() else ""
        opts.append(f'<option value="{_esc(v)}"{sel}>{_esc(v)}</option>')
    return "".join(opts)


def _render_filters(
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    mode: Optional[str],
    all_users: list[str],
    all_modes: list[str],
) -> str:
    return f"""
  <div class="card section">
    <h2 class="section-heading">Filters</h2>
    <form class="filters-form" method="get" action="/dashboard">
      <div class="field">
        <label for="start_date">Start date</label>
        <input type="date" id="start_date" name="start_date" value="{_esc(start_date)}">
      </div>
      <div class="field">
        <label for="end_date">End date</label>
        <input type="date" id="end_date" name="end_date" value="{_esc(end_date)}">
      </div>
      <div class="field">
        <label for="user_id">User</label>
        <select id="user_id" name="user_id">{_select_options(all_users, user_id)}</select>
      </div>
      <div class="field">
        <label for="session_id">Session ID</label>
        <input type="text" id="session_id" name="session_id" placeholder="e.g. test5-session" value="{_esc(session_id)}">
      </div>
      <div class="field">
        <label for="mode">Mode</label>
        <select id="mode" name="mode">{_select_options(all_modes, mode)}</select>
      </div>
      <div class="field field-actions">
        <button type="submit" class="btn btn-primary">Apply</button>
        <a href="/dashboard" class="btn btn-secondary">Reset</a>
      </div>
    </form>
  </div>"""


def _render_rollup_table(title: str, header_label: str, rows: list[dict]) -> str:
    if not rows:
        body = '<div class="empty-state">No data for the selected filters.</div>'
        return f"""
  <div class="section">
    <h2 class="section-heading">{html.escape(title)}</h2>
    <div class="table-wrap">{body}</div>
  </div>"""

    body_rows = []
    for r in rows:
        label = r["label"]
        label_html = (
            f'<span class="mode-badge">{_esc(label)}</span>' if header_label == "Mode" else _esc(label)
        )
        avg_time = r["time_seconds"] / r["requests"] if r["requests"] else 0.0
        body_rows.append(
            "<tr>"
            f"<td>{label_html}</td>"
            f"<td>{_fmt_int(r['requests'])}</td>"
            f"<td>{_fmt_int(r['input_tokens'] + r['output_tokens'])}</td>"
            f"<td>{_fmt_duration(r['time_seconds'])}</td>"
            f"<td>{_fmt_duration(avg_time)}</td>"
            f"<td>{_fmt_cost(r['cost_usd'])}</td>"
            "</tr>"
        )

    return f"""
  <div class="section">
    <h2 class="section-heading">{html.escape(title)}</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>{html.escape(header_label)}</th>
            <th>Requests</th>
            <th>Total Tokens</th>
            <th>Request Time</th>
            <th>Avg Time per Request</th>
            <th>Total Cost</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </div>"""


def _render_detail_table(rows: list[RequestRow]) -> str:
    if not rows:
        body = '<div class="empty-state">No data for the selected filters.</div>'
        return f"""
  <div class="section">
    <h2 class="section-heading">Detailed Breakdown</h2>
    <div class="table-wrap">{body}</div>
  </div>"""

    body_rows = []
    for r in rows:
        query_text = r.query or "—"
        body_rows.append(
            "<tr>"
            f"<td>{_esc(r.date)}</td>"
            f"<td>{_esc(r.user_id)}</td>"
            f"<td>{_esc(r.session_id)}</td>"
            f'<td><span class="mode-badge">{_esc(r.mode)}</span></td>'
            f'<td class="query-cell" title="{_esc(r.query)}">{_esc(query_text)}</td>'
            f"<td>{_fmt_int(r.tokens)}</td>"
            f"<td>{_fmt_duration(r.time_seconds)}</td>"
            f"<td>{_fmt_cost(r.cost_usd)}</td>"
            "</tr>"
        )

    return f"""
  <div class="section">
    <h2 class="section-heading">Detailed Breakdown</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>User</th>
            <th>Session</th>
            <th>Mode</th>
            <th>Query</th>
            <th>Total Tokens</th>
            <th>Request Time</th>
            <th>Total Cost</th>
          </tr>
        </thead>
        <tbody>
          {''.join(body_rows)}
        </tbody>
      </table>
    </div>
  </div>"""


def _render_page(
    *,
    groups: list[GroupTotals],
    request_rows: list[RequestRow],
    start_date: Optional[str],
    end_date: Optional[str],
    user_id: Optional[str],
    session_id: Optional[str],
    mode: Optional[str],
    all_users: list[str],
    all_modes: list[str],
) -> str:
    total_requests = sum(g.requests for g in groups)
    total_tokens = sum(g.total_tokens for g in groups)
    total_time = sum(g.time_seconds for g in groups)
    total_cost = sum(g.cost_usd for g in groups)

    kpi_html = "".join(
        [
            _kpi_card("Total Requests", _fmt_int(total_requests)),
            _kpi_card("Total Tokens", _fmt_int(total_tokens)),
            _kpi_card("Request Time", _fmt_duration(total_time)),
            _kpi_card("Total Cost", _fmt_cost(total_cost)),
        ]
    )

    by_day = _rollup(groups, lambda g: g.date)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AskEngen Analytics Dashboard</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
  <div class="page">
    <h1 class="main-heading">AskEngen Analytics Dashboard</h1>
    {_render_model_status()}
    {_render_filters(
        start_date=start_date, end_date=end_date, user_id=user_id,
        session_id=session_id, mode=mode, all_users=all_users, all_modes=all_modes,
    )}
    <div class="kpi-grid">{kpi_html}</div>
    {_render_rollup_table("Cost & Time by Day", "Date", by_day)}
    {_render_detail_table(request_rows)}
  </div>
</body>
</html>"""


# --------------------------------------------------------------------------
# Route
# --------------------------------------------------------------------------


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD, inclusive"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD, inclusive"),
    user_id: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    mode: Optional[str] = Query(None, description="Project name"),
) -> HTMLResponse:
    """Cost + time analytics, broken down by day, user, session, and mode.
    Reads directly off costs/*.jsonl and logs/*.jsonl (see
    src.utils.cost_utils and src.utils.logging_utils), plus any pre-migration
    costs/*.yml and logs/*.yml — no database involved, so this always
    reflects whatever has been flushed to disk, including today's
    still-open log file."""
    raw_cost_entries = _load_entries(COSTS_DIR)
    raw_log_entries = _load_entries(LOGS_DIR)

    all_users = sorted({e.get("user_id") or _UNKNOWN for e in raw_log_entries})
    all_modes = sorted({(e.get("mode") or _UNKNOWN) for e in raw_log_entries})

    filter_kwargs = dict(
        start_date=start_date, end_date=end_date, user_id=user_id, session_id=session_id, mode=mode
    )
    cost_entries = [e for e in raw_cost_entries if _matches_filters(e, **filter_kwargs)]
    log_entries = [e for e in raw_log_entries if _matches_filters(e, **filter_kwargs)]

    groups = _aggregate(cost_entries, log_entries)
    request_rows = _build_request_rows(cost_entries, log_entries)

    page = _render_page(
        groups=groups,
        request_rows=request_rows,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        session_id=session_id,
        mode=mode,
        all_users=all_users,
        all_modes=all_modes,
    )
    return HTMLResponse(content=page)
