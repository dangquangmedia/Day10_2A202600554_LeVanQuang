from __future__ import annotations

from datetime import UTC, date
from datetime import datetime
from html import unescape
import re
from typing import Any

import pandas as pd

from core.utils import compact_join, normalize_whitespace
from ingestion.crossref import PaperRecord


OUTPUT_COLUMNS = [
    "paper_id",
    "title",
    "summary",
    "authors",
    "categories",
    "primary_category",
    "published",
    "updated",
    "age_days",
    "authors_joined",
    "categories_joined",
    "summary_chars",
    "text_for_embedding",
    "abs_url",
    "pdf_url",
    "comment",
]


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Clean raw records into an embedding-ready dataframe."""
    run_day = _as_utc_date(run_date)
    rows: list[dict[str, Any]] = []

    for record in records:
        paper_id = _clean_text(record.paper_id).lower()
        title = _clean_text(record.title)
        summary = _clean_text(record.summary)
        authors = _clean_list(record.authors)
        categories = _clean_list(record.categories)
        published_date = _parse_date(record.published)
        updated_date = _parse_date(record.updated) or published_date

        if not _is_usable_record(paper_id, title, summary, authors, categories, published_date):
            continue

        published = published_date.isoformat()
        updated = updated_date.isoformat() if updated_date else published
        authors_joined = compact_join(authors)
        categories_joined = compact_join(categories)
        primary_category = _clean_text(record.primary_category) or categories[0]
        age_days = max(0, (run_day - published_date).days)

        rows.append(
            {
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": primary_category,
                "published": published,
                "updated": updated,
                "age_days": age_days,
                "authors_joined": authors_joined,
                "categories_joined": categories_joined,
                "summary_chars": len(summary),
                "text_for_embedding": _build_text_for_embedding(
                    title=title,
                    summary=summary,
                    authors_joined=authors_joined,
                    categories_joined=categories_joined,
                    published=published,
                ),
                "abs_url": _clean_text(record.abs_url),
                "pdf_url": _clean_text(record.pdf_url),
                "comment": _clean_text(record.comment),
            }
        )

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = pd.DataFrame(rows)
    df = (
        df.sort_values(["paper_id", "published"], ascending=[True, False])
        .drop_duplicates(subset=["paper_id"], keep="first")
        .sort_values(["published", "paper_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return df[OUTPUT_COLUMNS]


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", unescape(value or ""))
    return normalize_whitespace(text)


def _clean_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _clean_text(str(value))
        key = text.lower()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except ValueError:
        return None


def _as_utc_date(value: datetime) -> date:
    if value.tzinfo is None:
        return value.date()
    return value.astimezone(UTC).date()


def _is_usable_record(
    paper_id: str,
    title: str,
    summary: str,
    authors: list[str],
    categories: list[str],
    published: date | None,
) -> bool:
    return bool(
        paper_id
        and len(title) >= 8
        and len(summary) >= 40
        and authors
        and categories
        and published
    )


def _build_text_for_embedding(
    *,
    title: str,
    summary: str,
    authors_joined: str,
    categories_joined: str,
    published: str,
) -> str:
    return "\n".join(
        [
            f"Title: {title}",
            f"Summary: {summary}",
            f"Authors: {authors_joined}",
            f"Categories: {categories_joined}",
            f"Published: {published}",
        ]
    )
