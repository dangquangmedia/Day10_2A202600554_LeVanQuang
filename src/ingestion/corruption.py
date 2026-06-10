from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.utils import write_json


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Simulate realistic data quality issues in a cleaned dataframe."""
    corrupted = df.copy().reset_index(drop=True)
    log: list[dict] = []

    if corrupted.empty:
        write_json(Path(output_log_path), {"total_input_rows": 0, "total_output_rows": 0, "operations": log})
        return corrupted

    latest_count = min(max(1, len(corrupted) // 8), max(1, len(corrupted) - 1))
    latest_ids = _latest_indices(corrupted, latest_count)
    dropped_ids = corrupted.loc[latest_ids, "paper_id"].astype(str).tolist()
    if len(corrupted) > 2:
        corrupted = corrupted.drop(index=latest_ids).reset_index(drop=True)
        log.append({"operation": "drop_latest_records", "rows": len(dropped_ids), "paper_ids": dropped_ids})
    else:
        log.append({"operation": "drop_latest_records_skipped", "reason": "dataset too small"})

    if not corrupted.empty:
        blank_idx = corrupted.index[: max(1, len(corrupted) // 6)].tolist()
        blank_ids = corrupted.loc[blank_idx, "paper_id"].astype(str).tolist()
        corrupted.loc[blank_idx, "summary"] = ""
        log.append({"operation": "blank_summary", "rows": len(blank_idx), "paper_ids": blank_ids})

    if len(corrupted) >= 2:
        noise_idx = corrupted.index[-max(1, len(corrupted) // 6) :].tolist()
        noise_ids = corrupted.loc[noise_idx, "paper_id"].astype(str).tolist()
        corrupted.loc[noise_idx, "summary"] = (
            corrupted.loc[noise_idx, "summary"].fillna("").astype(str)
            + " NOISE_TOKEN NOISE_TOKEN unrelated corrupted retrieval text."
        )
        log.append({"operation": "inject_summary_noise", "rows": len(noise_idx), "paper_ids": noise_ids})

    if len(corrupted) >= 3:
        title_idx = corrupted.index[1 : 1 + max(1, len(corrupted) // 8)].tolist()
        title_ids = corrupted.loc[title_idx, "paper_id"].astype(str).tolist()
        corrupted.loc[title_idx, "title"] = corrupted.loc[title_idx, "title"].fillna("").astype(str).str.slice(0, 18)
        log.append({"operation": "truncate_title", "rows": len(title_idx), "paper_ids": title_ids})

    if "age_days" in corrupted and not corrupted.empty:
        stale_idx = corrupted.index[-max(1, len(corrupted) // 5) :].tolist()
        stale_ids = corrupted.loc[stale_idx, "paper_id"].astype(str).tolist()
        corrupted.loc[stale_idx, "published"] = "2018-01-01"
        corrupted.loc[stale_idx, "updated"] = "2018-01-02"
        corrupted.loc[stale_idx, "age_days"] = 9999
        log.append({"operation": "make_publication_date_stale", "rows": len(stale_idx), "paper_ids": stale_ids})

    if not corrupted.empty:
        duplicate_count = min(max(1, len(corrupted) // 8), len(corrupted))
        duplicates = corrupted.head(duplicate_count).copy()
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        log.append(
            {
                "operation": "add_duplicate_rows",
                "rows": int(len(duplicates)),
                "paper_ids": duplicates["paper_id"].astype(str).tolist(),
            }
        )

    corrupted = _rebuild_derived_columns(corrupted)
    write_json(
        Path(output_log_path),
        {
            "total_input_rows": int(len(df)),
            "total_output_rows": int(len(corrupted)),
            "operations": log,
        },
    )
    return corrupted


def _latest_indices(df: pd.DataFrame, count: int) -> list[int]:
    if "published" not in df:
        return df.index[:count].tolist()
    published = pd.to_datetime(df["published"], errors="coerce")
    return published.sort_values(ascending=False).head(count).index.tolist()


def _rebuild_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    rebuilt = df.copy()
    if "summary" in rebuilt:
        rebuilt["summary"] = rebuilt["summary"].fillna("").astype(str)
        rebuilt["summary_chars"] = rebuilt["summary"].str.len()
    if "title" in rebuilt:
        rebuilt["title"] = rebuilt["title"].fillna("").astype(str)
    if "authors_joined" not in rebuilt and "authors" in rebuilt:
        rebuilt["authors_joined"] = rebuilt["authors"].apply(_join_if_list)
    if "categories_joined" not in rebuilt and "categories" in rebuilt:
        rebuilt["categories_joined"] = rebuilt["categories"].apply(_join_if_list)

    if "authors_joined" not in rebuilt:
        rebuilt["authors_joined"] = ""
    if "categories_joined" not in rebuilt:
        rebuilt["categories_joined"] = ""
    if "published" not in rebuilt:
        rebuilt["published"] = ""
    rebuilt["authors_joined"] = rebuilt["authors_joined"].fillna("").astype(str)
    rebuilt["categories_joined"] = rebuilt["categories_joined"].fillna("").astype(str)
    rebuilt["published"] = rebuilt["published"].fillna("").astype(str)
    rebuilt["text_for_embedding"] = rebuilt.apply(_embedding_text, axis=1)
    return rebuilt


def _join_if_list(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    return str(value or "")


def _embedding_text(row: pd.Series) -> str:
    return "\n".join(
        [
            f"Title: {row.get('title', '')}",
            f"Summary: {row.get('summary', '')}",
            f"Authors: {row.get('authors_joined', '')}",
            f"Categories: {row.get('categories_joined', '')}",
            f"Published: {row.get('published', '')}",
        ]
    )
