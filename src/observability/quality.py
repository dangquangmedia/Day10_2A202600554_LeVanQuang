from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import write_json


REQUIRED_COLUMNS = [
    "paper_id",
    "title",
    "summary",
    "published",
    "updated",
    "age_days",
    "authors_joined",
    "categories_joined",
    "text_for_embedding",
]


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Run data quality gates and persist a JSON report."""
    checks = [
        _check("row_count_positive", "volume", len(df) > 0, len(df), "> 0"),
        _check("required_schema", "schema", all(column in df.columns for column in REQUIRED_COLUMNS), sorted(df.columns), REQUIRED_COLUMNS),
    ]

    if len(df) == 0:
        checks.extend(
            [
                _check("paper_id_not_empty", "completeness", False, 0, "0 empty values"),
                _check("paper_id_unique", "uniqueness", False, 0, "unique paper_id"),
                _check("title_not_empty", "completeness", False, 0, "0 empty values"),
                _check("summary_not_empty", "completeness", False, 0, "0 empty values"),
                _check("summary_length", "validity", False, 0, "min >= 40 chars"),
                _check("freshness_threshold", "timeliness", False, 0, f"age_days <= {settings.freshness_threshold_days}"),
            ]
        )
    else:
        checks.extend(_non_empty_checks(df))
        checks.append(
            _check(
                "paper_id_unique",
                "uniqueness",
                bool(df["paper_id"].is_unique) if "paper_id" in df else False,
                int(df["paper_id"].duplicated().sum()) if "paper_id" in df else "missing",
                "0 duplicate paper_id",
            )
        )
        checks.append(_summary_length_check(df))
        checks.append(_freshness_check(df, settings))
        checks.append(_summary_distribution_check(df))

    failed = [check for check in checks if not check["passed"]]
    report = {
        "report_name": report_name,
        "total_rows": int(len(df)),
        "total_checks": len(checks),
        "failed_count": len(failed),
        "passed": len(failed) == 0,
        "checks": checks,
    }
    write_json(settings.paths.quality_dir / f"{report_name}.json", report)
    return report


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Build and persist a freshness summary report."""
    if "published" in df and len(df) > 0:
        published = pd.to_datetime(df["published"], errors="coerce").dropna()
        latest = published.max().date().isoformat() if not published.empty else None
        oldest = published.min().date().isoformat() if not published.empty else None
    else:
        latest = None
        oldest = None

    stale_rows = _stale_row_count(df, settings)
    report = {
        "latest_published": latest,
        "oldest_published": oldest,
        "stale_rows": stale_rows,
        "total_rows": int(len(df)),
        "freshness_threshold_days": settings.freshness_threshold_days,
        "is_fresh": stale_rows == 0 and len(df) > 0,
    }
    write_json(Path(report_path), report)
    return report


def _check(name: str, dimension: str, passed: bool, observed: Any, threshold: Any) -> dict[str, Any]:
    return {
        "name": name,
        "dimension": dimension,
        "passed": bool(passed),
        "observed": observed,
        "threshold": threshold,
    }


def _non_empty_checks(df: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for column, dimension in [
        ("paper_id", "completeness"),
        ("title", "completeness"),
        ("summary", "completeness"),
        ("published", "validity"),
        ("authors_joined", "completeness"),
        ("categories_joined", "completeness"),
        ("text_for_embedding", "completeness"),
    ]:
        if column not in df:
            checks.append(_check(f"{column}_not_empty", dimension, False, "missing", "0 empty values"))
            continue
        empty_count = int(df[column].fillna("").astype(str).str.strip().eq("").sum())
        checks.append(_check(f"{column}_not_empty", dimension, empty_count == 0, empty_count, "0 empty values"))
    return checks


def _summary_length_check(df: pd.DataFrame) -> dict[str, Any]:
    if "summary" not in df:
        return _check("summary_length", "validity", False, "missing", "min >= 40 chars")
    lengths = df["summary"].fillna("").astype(str).str.len()
    minimum = int(lengths.min()) if len(lengths) else 0
    return _check("summary_length", "validity", minimum >= 40, minimum, "min >= 40 chars")


def _freshness_check(df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    stale_rows = _stale_row_count(df, settings)
    return _check(
        "freshness_threshold",
        "timeliness",
        stale_rows == 0,
        stale_rows,
        f"age_days <= {settings.freshness_threshold_days}",
    )


def _summary_distribution_check(df: pd.DataFrame) -> dict[str, Any]:
    if "summary" not in df:
        return _check("summary_length_distribution", "distribution", False, "missing", "mean >= 80 chars")
    lengths = df["summary"].fillna("").astype(str).str.len()
    observed = {
        "min": int(lengths.min()) if len(lengths) else 0,
        "mean": round(float(lengths.mean()), 2) if len(lengths) else 0.0,
        "max": int(lengths.max()) if len(lengths) else 0,
    }
    return _check("summary_length_distribution", "distribution", observed["mean"] >= 60, observed, "mean >= 60 chars")


def _stale_row_count(df: pd.DataFrame, settings: Settings) -> int:
    if "age_days" not in df:
        return int(len(df))
    ages = pd.to_numeric(df["age_days"], errors="coerce")
    return int((ages > settings.freshness_threshold_days).fillna(True).sum())
