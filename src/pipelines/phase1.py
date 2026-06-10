from __future__ import annotations

from core.config import load_settings
from core.utils import now_utc, write_csv, write_json, write_text
from evaluation.metrics import evaluate_pipeline
from evaluation.testset import build_test_set
from ingestion.cleaning import build_clean_dataframe
from ingestion.crossref import fetch_source_records, load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_phase1_report
from retrieval.index import LocalEmbeddingIndex


def main() -> None:
    """Run the baseline data pipeline end-to-end."""
    settings = load_settings()

    if settings.refresh_source or not settings.paths.raw_records_json.exists():
        records = fetch_source_records(settings)
    else:
        records = load_raw_records(settings.paths.raw_records_json)

    df = build_clean_dataframe(records, run_date=now_utc())
    write_csv(df, settings.paths.clean_csv)
    write_json(settings.paths.clean_json, df.to_dict(orient="records"))

    index = LocalEmbeddingIndex.build(
        df,
        settings,
        embeddings_output_path=settings.paths.embeddings_json,
    )

    if settings.refresh_test_set or not settings.paths.eval_testset.exists():
        build_test_set(df, settings.paths.eval_testset)

    evaluation = evaluate_pipeline(
        settings=settings,
        index=index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.baseline_metrics,
        answers_output_path=settings.paths.baseline_answers,
    )
    quality = run_data_quality_checks(df, settings, "baseline_quality")
    freshness = build_freshness_report(df, settings, settings.paths.freshness_report)
    generate_phase1_report(
        settings.paths.baseline_report,
        source_summary={
            "source_api": settings.source_api,
            "source_query": settings.source_query,
            "source_filter": settings.source_filter,
            "max_results": settings.max_results,
            "raw_records": len(records),
            "clean_records": int(len(df)),
            "raw_records_path": str(settings.paths.raw_records_json),
            "clean_path": str(settings.paths.clean_json),
            "embeddings_path": str(settings.paths.embeddings_json),
        },
        metrics=evaluation.summary,
        quality=quality,
        freshness=freshness,
    )
