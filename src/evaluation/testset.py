from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.utils import first_sentence, write_json


REQUIRED_COLUMNS = {
    "paper_id",
    "title",
    "summary",
    "authors_joined",
    "categories_joined",
    "published",
}


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Create a deterministic evaluation set from the cleaned paper dataframe."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Clean dataframe is missing required columns: {', '.join(missing)}")

    samples = _select_samples(df)
    if len(samples) < 2:
        raise ValueError("Evaluation set requires at least 2 usable cleaned records.")

    test_set: list[dict[str, Any]] = []
    for sample_index, row in enumerate(samples.to_dict(orient="records")):
        paper_id = str(row["paper_id"])
        title = str(row["title"])
        doc_ids = [paper_id]
        test_set.extend(
            [
                _item(
                    sample_index=sample_index,
                    question_type="summary",
                    question=f"What is the main contribution of '{title}'?",
                    ground_truth=first_sentence(str(row["summary"])),
                    doc_ids=doc_ids,
                ),
                _item(
                    sample_index=sample_index,
                    question_type="authors",
                    question=f"Who authored '{title}'?",
                    ground_truth=str(row["authors_joined"]),
                    doc_ids=doc_ids,
                ),
                _item(
                    sample_index=sample_index,
                    question_type="date",
                    question=f"When was '{title}' published?",
                    ground_truth=str(row["published"]),
                    doc_ids=doc_ids,
                ),
                _item(
                    sample_index=sample_index,
                    question_type="categories",
                    question=f"What categories are associated with '{title}'?",
                    ground_truth=str(row["categories_joined"]),
                    doc_ids=doc_ids,
                ),
            ]
        )

    write_json(Path(output_path), test_set)
    return test_set


def _select_samples(df: pd.DataFrame) -> pd.DataFrame:
    usable = df.copy()
    for column in REQUIRED_COLUMNS:
        usable = usable[usable[column].astype(str).str.strip() != ""]
    usable = usable[usable["summary"].astype(str).str.len() >= 40]
    usable = usable.sort_values(["published", "paper_id"], ascending=[False, True])
    return usable.head(8).reset_index(drop=True)


def _item(
    *,
    sample_index: int,
    question_type: str,
    question: str,
    ground_truth: str,
    doc_ids: list[str],
) -> dict[str, Any]:
    return {
        "id": f"{question_type}-{sample_index + 1:03d}",
        "question_type": question_type,
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_doc_ids": doc_ids,
    }
