from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.config import load_settings
from core.utils import read_json
from ingestion.crossref import PaperRecord
from pipelines import phase1


def _settings(tmp_path: Path):
    settings = load_settings()
    paths = replace(
        settings.paths,
        raw_api_response=tmp_path / "raw" / "crossref_response.json",
        raw_records_json=tmp_path / "raw" / "crossref_records.json",
        clean_csv=tmp_path / "clean" / "papers_clean.csv",
        clean_json=tmp_path / "clean" / "papers_clean.json",
        chroma_dir=tmp_path / "chroma",
        embeddings_json=tmp_path / "embeddings" / "papers_embeddings.json",
        eval_testset=tmp_path / "eval" / "test_set.json",
        baseline_metrics=tmp_path / "results" / "baseline_metrics.json",
        baseline_answers=tmp_path / "results" / "baseline_answers.json",
        quality_dir=tmp_path / "quality",
        freshness_report=tmp_path / "quality" / "freshness_report.json",
        baseline_report=tmp_path / "reports" / "phase1_report.md",
    )
    return replace(settings, paths=paths, refresh_source=False, refresh_test_set=False)


def _record() -> PaperRecord:
    return PaperRecord(
        paper_id="10.1234/example",
        title="Reliable RAG Pipeline",
        summary="This paper explains reliable data pipelines for RAG systems.",
        authors=["Ada Lovelace"],
        categories=["AI"],
        primary_category="AI",
        published="2025-05-01",
        updated="2025-05-02",
        abs_url="https://doi.org/10.1234/example",
        pdf_url="",
        comment="",
    )


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper_id": "10.1234/example",
                "title": "Reliable RAG Pipeline",
                "summary": "This paper explains reliable data pipelines for RAG systems.",
                "authors": ["Ada Lovelace"],
                "categories": ["AI"],
                "primary_category": "AI",
                "published": "2025-05-01",
                "updated": "2025-05-02",
                "age_days": 40,
                "authors_joined": "Ada Lovelace",
                "categories_joined": "AI",
                "summary_chars": 61,
                "text_for_embedding": "Title: Reliable RAG Pipeline",
                "abs_url": "https://doi.org/10.1234/example",
                "pdf_url": "",
                "comment": "",
            }
        ]
    )


def test_phase1_main_runs_baseline_flow(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(phase1, "load_settings", lambda: settings)
    monkeypatch.setattr(phase1, "fetch_source_records", lambda settings: calls.append("fetch") or [_record()])
    monkeypatch.setattr(phase1, "load_raw_records", lambda path: calls.append("load_raw") or [_record()])
    monkeypatch.setattr(phase1, "build_clean_dataframe", lambda records, run_date: calls.append("clean") or _df())

    class FakeIndex:
        @classmethod
        def build(cls, df, settings, embeddings_output_path=None):
            calls.append("index")
            return cls()

    monkeypatch.setattr(phase1, "LocalEmbeddingIndex", FakeIndex)

    def fake_build_test_set(df, output_path):
        calls.append("testset")
        payload = [
            {
                "id": "summary-001",
                "question_type": "summary",
                "question": "What is the main contribution?",
                "ground_truth": "Reliable pipelines.",
                "ground_truth_doc_ids": ["10.1234/example"],
            }
        ]
        phase1.write_json(output_path, payload)
        return payload

    monkeypatch.setattr(phase1, "build_test_set", fake_build_test_set)
    monkeypatch.setattr(
        phase1,
        "evaluate_pipeline",
        lambda **kwargs: calls.append("evaluate")
        or SimpleNamespace(summary={"samples": 1, "retrieval_hit_rate": 1.0}, answers=[]),
    )
    monkeypatch.setattr(
        phase1,
        "run_data_quality_checks",
        lambda df, settings, report_name: calls.append("quality") or {"passed": True, "checks": []},
    )
    monkeypatch.setattr(
        phase1,
        "build_freshness_report",
        lambda df, settings, report_path: calls.append("freshness") or {"is_fresh": True},
    )
    monkeypatch.setattr(
        phase1,
        "generate_phase1_report",
        lambda report_path, source_summary, metrics, quality, freshness: calls.append("report")
        or phase1.write_text(report_path, "# report\n"),
    )

    phase1.main()

    assert calls == ["fetch", "clean", "index", "testset", "evaluate", "quality", "freshness", "report"]
    assert settings.paths.clean_csv.exists()
    assert read_json(settings.paths.clean_json)[0]["paper_id"] == "10.1234/example"
    assert read_json(settings.paths.eval_testset)[0]["id"] == "summary-001"
    assert settings.paths.baseline_report.read_text(encoding="utf-8") == "# report\n"
