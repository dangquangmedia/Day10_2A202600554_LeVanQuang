from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from core.config import load_settings
from core.utils import read_json
from observability.quality import build_freshness_report, run_data_quality_checks


def _settings(tmp_path: Path):
    settings = load_settings()
    paths = replace(settings.paths, quality_dir=tmp_path / "quality", freshness_report=tmp_path / "quality" / "fresh.json")
    return replace(settings, paths=paths, freshness_threshold_days=180)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "10.1/a",
                "title": "Fresh paper",
                "summary": "A complete abstract with enough detail for retrieval and quality validation.",
                "published": "2025-06-01",
                "updated": "2025-06-02",
                "age_days": 9,
                "authors_joined": "Ada Lovelace",
                "categories_joined": "AI",
                "text_for_embedding": "Title: Fresh paper",
            },
            {
                "paper_id": "10.1/b",
                "title": "Older paper",
                "summary": "Another complete abstract with enough detail for retrieval and quality validation.",
                "published": "2025-03-15",
                "updated": "2025-03-16",
                "age_days": 87,
                "authors_joined": "Grace Hopper",
                "categories_joined": "Data Engineering",
                "text_for_embedding": "Title: Older paper",
            },
        ]
    )


def _stale_df() -> pd.DataFrame:
    df = _df()
    df.loc[1, "published"] = "2024-12-01"
    df.loc[1, "updated"] = "2024-12-02"
    df.loc[1, "age_days"] = 191
    return df


def test_run_data_quality_checks_writes_dimensioned_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    report = run_data_quality_checks(_df(), settings, "baseline_quality")

    assert report["report_name"] == "baseline_quality"
    assert report["total_rows"] == 2
    assert report["total_checks"] >= 8
    assert report["failed_count"] == 0
    assert report["passed"] is True
    assert {"completeness", "validity", "uniqueness", "timeliness", "volume", "schema", "distribution"} <= {
        check["dimension"] for check in report["checks"]
    }
    assert read_json(settings.paths.quality_dir / "baseline_quality.json") == report


def test_run_data_quality_checks_flags_duplicates_and_stale_data(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    bad_df = _df()
    bad_df.loc[1, "paper_id"] = "10.1/a"
    bad_df.loc[0, "summary"] = ""
    bad_df.loc[1, "age_days"] = 191

    report = run_data_quality_checks(bad_df, settings, "bad_quality")

    failed_names = {check["name"] for check in report["checks"] if not check["passed"]}
    assert report["passed"] is False
    assert "paper_id_unique" in failed_names
    assert "summary_not_empty" in failed_names
    assert "freshness_threshold" in failed_names


def test_build_freshness_report_writes_summary(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    output_path = tmp_path / "quality" / "freshness.json"

    report = build_freshness_report(_stale_df(), settings, output_path)

    assert report == {
        "latest_published": "2025-06-01",
        "oldest_published": "2024-12-01",
        "stale_rows": 1,
        "total_rows": 2,
        "freshness_threshold_days": 180,
        "is_fresh": False,
    }
    assert read_json(output_path) == report
