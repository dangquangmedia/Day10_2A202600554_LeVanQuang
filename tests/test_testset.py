from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.utils import read_json
from evaluation.testset import build_test_set


def _clean_df(rows: int = 3) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "paper_id": f"10.1234/paper-{index}",
                "title": f"Reliable RAG Pipeline {index}",
                "summary": (
                    f"Paper {index} explains how data quality, metadata, and freshness "
                    "improve retrieval augmented generation systems."
                ),
                "authors": [f"Author {index}", "Ada Lovelace"],
                "categories": ["Artificial Intelligence", "Data Engineering"],
                "primary_category": "Artificial Intelligence",
                "published": f"2025-05-{index + 1:02d}",
                "updated": f"2025-06-{index + 1:02d}",
                "age_days": 30 + index,
                "authors_joined": f"Author {index}, Ada Lovelace",
                "categories_joined": "Artificial Intelligence, Data Engineering",
                "summary_chars": 110,
                "text_for_embedding": f"Title: Reliable RAG Pipeline {index}",
                "abs_url": f"https://doi.org/10.1234/paper-{index}",
                "pdf_url": "",
                "comment": "",
            }
        )
    return pd.DataFrame(records)


def test_build_test_set_creates_question_types_and_writes_json(tmp_path: Path) -> None:
    output_path = tmp_path / "test_set.json"

    test_set = build_test_set(_clean_df(), output_path)

    assert output_path.exists()
    assert read_json(output_path) == test_set
    assert {item["question_type"] for item in test_set} == {"summary", "authors", "date", "categories"}
    assert len(test_set) == 12
    assert len({item["id"] for item in test_set}) == len(test_set)
    assert all(set(item) == {"id", "question_type", "question", "ground_truth", "ground_truth_doc_ids"} for item in test_set)
    assert test_set[0]["question"] == "What is the main contribution of 'Reliable RAG Pipeline 2'?"
    assert test_set[0]["ground_truth"].startswith("Paper 2 explains")
    assert test_set[0]["ground_truth_doc_ids"] == ["10.1234/paper-2"]
    assert any(item["ground_truth"] == "Author 0, Ada Lovelace" for item in test_set)
    assert any(item["ground_truth"] == "2025-05-01" for item in test_set)
    assert any(item["ground_truth"] == "Artificial Intelligence, Data Engineering" for item in test_set)


def test_build_test_set_rejects_too_small_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 2"):
        build_test_set(_clean_df(rows=1), tmp_path / "test_set.json")
