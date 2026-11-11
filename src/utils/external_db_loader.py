"""External source registry and URL content fetcher.

Each registered source has:
  key_name    — short identifier the user and agent refer to
  description — human-readable label describing what the source contains
  link        — public URL (web page, direct Excel/CSV/PDF link)

At query time the agent fetches the URL live and passes the extracted
content to the LLM — no local file storage required.
"""

import io
import json
import re
from html.parser import HTMLParser
from pathlib import Path

from src.config import DATA_DIR

_CONTENT_LIMIT = 8_000   # max characters of extracted text sent to LLM
_TABLE_ROW_LIMIT = 200   # max rows from Excel/CSV converted to markdown

# External sources are always stored under the fixed analytics namespace
# regardless of what project_name the caller passes, so the query orchestrator
# (which runs with mode="analytics") always finds them.
_NS = "analytics"


def _externaldb_path(project: str = _NS) -> Path:  # noqa: ARG001
    return DATA_DIR / _NS / "externaldb.json"


def load_external_db(project: str) -> list[dict]:
    path = _externaldb_path(project)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_external_db(project: str, sources: list[dict]) -> None:
    path = _externaldb_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2, ensure_ascii=False)


def get_source_by_key(project: str, key_name: str) -> dict | None:
    for src in load_external_db(project):
        if src.get("key_name") == key_name:
            return src
    return None


class _TextExtractor(HTMLParser):
    """Strips HTML tags and collapses whitespace to plain text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "nav", "footer", "header"}:
            self._skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "nav", "footer", "header"}:
            self._skip = False
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr"}:
            self._parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of whitespace/newlines
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _df_to_markdown(df) -> str:
    """Converts a DataFrame to a compact markdown table string."""
    rows = df.head(_TABLE_ROW_LIMIT)
    header = " | ".join(str(c) for c in rows.columns)
    sep = " | ".join("---" for _ in rows.columns)
    lines = [header, sep]
    for _, row in rows.iterrows():
        lines.append(" | ".join(str(v) for v in row))
    return "\n".join(lines)


async def fetch_url_content(url: str) -> str:
    """Fetch a URL and return its content as plain text.

    Supports:
      - .xlsx / .xls  → pandas → markdown table (first 200 rows)
      - .csv          → pandas → markdown table (first 200 rows)
      - .pdf          → pypdf text extraction
      - HTML page     → tag-stripped plain text (truncated to 8000 chars)
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is not installed. Run: pip install httpx") from exc

    url_path = url.lower().split("?")[0]

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # ── Excel ────────────────────────────────────────────────────────────────
    if url_path.endswith((".xlsx", ".xls")) or "spreadsheet" in content_type:
        try:
            import pandas as pd
            xls = pd.ExcelFile(io.BytesIO(response.content))
            parts = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                parts.append(f"## Sheet: {sheet}\n{_df_to_markdown(df)}")
            return "\n\n".join(parts)[:_CONTENT_LIMIT]
        except Exception as exc:
            return f"[Could not parse Excel file: {exc}]"

    # ── CSV ──────────────────────────────────────────────────────────────────
    if url_path.endswith(".csv") or "text/csv" in content_type:
        try:
            import pandas as pd
            df = pd.read_csv(io.BytesIO(response.content))
            return _df_to_markdown(df)[:_CONTENT_LIMIT]
        except Exception as exc:
            return f"[Could not parse CSV file: {exc}]"

    # ── PDF ──────────────────────────────────────────────────────────────────
    if url_path.endswith(".pdf") or "application/pdf" in content_type:
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(response.content))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)[:_CONTENT_LIMIT]
        except Exception as exc:
            return f"[Could not parse PDF: {exc}]"

    # ── HTML (default) ───────────────────────────────────────────────────────
    try:
        html = response.text
        extractor = _TextExtractor()
        extractor.feed(html)
        return extractor.get_text()[:_CONTENT_LIMIT]
    except Exception as exc:
        return f"[Could not extract text from page: {exc}]"
