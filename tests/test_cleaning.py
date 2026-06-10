from __future__ import annotations

from datetime import UTC, datetime

from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import PaperRecord


def _record(
    paper_id: str = "10.1234/example",
    title: str = "  Agentic RAG With Data Quality  ",
    summary: str = "This paper studies reliable data pipelines for retrieval augmented generation systems.",
    authors: list[str] | None = None,
    categories: list[str] | None = None,
    published: str = "2025-05-09",
    updated: str = "2025-06-01",
) -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        summary=summary,
        authors=authors if authors is not None else ["Ada Lovelace", "Grace Hopper"],
        categories=categories if categories is not None else ["Artificial Intelligence", "Data Engineering"],
        primary_category=(categories or ["Artificial Intelligence"])[0] if categories != [] else "",
        published=published,
        updated=updated,
        abs_url=f"https://doi.org/{paper_id}",
        pdf_url="https://example.org/paper.pdf",
        comment="",
    )


def test_build_clean_dataframe_creates_embedding_ready_schema() -> None:
    df = build_clean_dataframe([_record()], run_date=datetime(2025, 6, 10, tzinfo=UTC))

    assert list(df.columns) == [
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
    row = df.iloc[0].to_dict()
    assert row["paper_id"] == "10.1234/example"
    assert row["title"] == "Agentic RAG With Data Quality"
    assert row["published"] == "2025-05-09"
    assert row["updated"] == "2025-06-01"
    assert row["age_days"] == 32
    assert row["authors_joined"] == "Ada Lovelace, Grace Hopper"
    assert row["categories_joined"] == "Artificial Intelligence, Data Engineering"
    assert row["summary_chars"] == len(row["summary"])
    assert "Title: Agentic RAG With Data Quality" in row["text_for_embedding"]
    assert "Authors: Ada Lovelace, Grace Hopper" in row["text_for_embedding"]
    assert "Published: 2025-05-09" in row["text_for_embedding"]


def test_build_clean_dataframe_deduplicates_by_paper_id_and_sorts() -> None:
    duplicate = _record(
        paper_id="10.1234/example",
        title="Duplicate should be removed",
        published="2025-05-10",
    )
    older = _record(paper_id="10.2222/older", title="Older Paper", published="2024-12-31", updated="2025-01-01")

    df = build_clean_dataframe([duplicate, older, _record()], run_date=datetime(2025, 6, 10, tzinfo=UTC))

    assert df["paper_id"].tolist() == ["10.1234/example", "10.2222/older"]
    assert df.loc[df["paper_id"] == "10.1234/example", "title"].item() == "Duplicate should be removed"


def test_build_clean_dataframe_filters_bad_records() -> None:
    records = [
        _record(paper_id="10.good/record"),
        _record(paper_id="10.bad/short-summary", summary="Too short."),
        _record(paper_id="10.bad/no-author", authors=[]),
        _record(paper_id="10.bad/no-category", categories=[]),
        _record(paper_id="10.bad/date", published="not-a-date"),
    ]

    df = build_clean_dataframe(records, run_date=datetime(2025, 6, 10, tzinfo=UTC))

    assert df["paper_id"].tolist() == ["10.good/record"]
