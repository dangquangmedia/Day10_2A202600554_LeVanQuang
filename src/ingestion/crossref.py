from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
import re
import time
from typing import Any

import requests

from core.config import Settings
from core.utils import normalize_whitespace, read_json, write_json


CROSSREF_API_URL = "https://api.crossref.org/works"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: str
    updated: str
    abs_url: str
    pdf_url: str
    comment: str


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse Crossref works payload into the stable raw record schema."""
    records: list[PaperRecord] = []
    for item in payload.get("message", {}).get("items", []):
        paper_id = _clean_text(str(item.get("DOI", ""))).lower()
        title = _first_text(item.get("title"))
        summary = _clean_abstract(str(item.get("abstract", "")))
        published = _extract_date(
            item.get("published-print")
            or item.get("published-online")
            or item.get("published")
            or item.get("issued")
            or item.get("created")
        )

        if not paper_id or not title or not summary or not published:
            continue

        categories = _clean_text_list(item.get("subject"))
        if not categories:
            fallback = item.get("group-title")
            if not fallback and item.get("short-container-title"):
                fallback = item.get("short-container-title")[0]
            if not fallback and item.get("container-title"):
                fallback = item.get("container-title")[0]
            
            if fallback:
                categories = _clean_text_list([fallback])
            else:
                categories = ["Computer Science"]

        authors = _extract_authors(item.get("author"))
        updated = _extract_date(item.get("updated")) or published

        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                primary_category=categories[0] if categories else "",
                published=published,
                updated=updated,
                abs_url=_clean_text(str(item.get("URL", ""))),
                pdf_url=_extract_pdf_url(item.get("link")),
                comment=_clean_text(str(item.get("short-container-title", [""])[0] if item.get("short-container-title") else "")),
            )
        )
    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Fetch Crossref source data, persist raw payload, and persist parsed records."""
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
    }
    response = None
    for attempt in range(3):
        response = requests.get(CROSSREF_API_URL, params=params, timeout=30)
        if response.status_code not in RETRY_STATUS_CODES:
            break
        if attempt < 2:
            time.sleep(2**attempt)

    if response is None:
        raise RuntimeError("Crossref request was not attempted.")
    response.raise_for_status()

    payload = response.json()
    write_json(settings.paths.raw_api_response, payload)
    records = parse_crossref_payload(payload)
    write_json(settings.paths.raw_records_json, [asdict(record) for record in records])
    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Load parsed raw records from a JSON snapshot."""
    payload = read_json(path)
    return [
        PaperRecord(
            paper_id=str(item.get("paper_id", "")),
            title=str(item.get("title", "")),
            summary=str(item.get("summary", "")),
            authors=list(item.get("authors") or []),
            categories=list(item.get("categories") or []),
            primary_category=str(item.get("primary_category", "")),
            published=str(item.get("published", "")),
            updated=str(item.get("updated", "")),
            abs_url=str(item.get("abs_url", "")),
            pdf_url=str(item.get("pdf_url", "")),
            comment=str(item.get("comment", "")),
        )
        for item in payload
    ]


def _clean_text(value: str) -> str:
    return normalize_whitespace(unescape(value or ""))


def _clean_abstract(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", unescape(value or ""))
    return _clean_text(text)


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return _clean_text(str(value[0])) if value else ""
    return _clean_text(str(value or ""))


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in value:
        text = _clean_text(str(item))
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
    return cleaned


def _extract_authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for author in value:
        if not isinstance(author, dict):
            continue
        name = _clean_text(str(author.get("name", "")))
        if not name:
            name = _clean_text(f"{author.get('given', '')} {author.get('family', '')}")
        if name:
            authors.append(name)
    return authors


def _extract_date(value: Any) -> str:
    if isinstance(value, dict):
        date_parts = value.get("date-parts")
        if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list):
            parts = [int(part) for part in date_parts[0] if part is not None]
            if not parts:
                return ""
            year = parts[0]
            month = parts[1] if len(parts) > 1 else 1
            day = parts[2] if len(parts) > 2 else 1
            return f"{year:04d}-{month:02d}-{day:02d}"
        timestamp = value.get("date-time")
        if isinstance(timestamp, str) and len(timestamp) >= 10:
            return timestamp[:10]
    if isinstance(value, str) and len(value) >= 10:
        return value[:10]
    return ""


def _extract_pdf_url(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for link in value:
        if not isinstance(link, dict):
            continue
        url = _clean_text(str(link.get("URL", "")))
        content_type = str(link.get("content-type", "")).lower()
        if url and ("pdf" in content_type or url.lower().endswith(".pdf")):
            return url
    return ""
