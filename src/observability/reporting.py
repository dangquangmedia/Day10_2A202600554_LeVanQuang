from __future__ import annotations

from typing import Any

from core.utils import write_text


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    """Write a readable markdown report for the baseline pipeline."""
    lines = [
        "# Phase 1 Baseline Report",
        "",
        "## Source",
        f"- Source API: {source_summary.get('source_api', '')}",
        f"- Query: {source_summary.get('source_query', '')}",
        f"- Filter: {source_summary.get('source_filter', '')}",
        f"- Raw records: {source_summary.get('raw_records', 0)}",
        f"- Clean records: {source_summary.get('clean_records', 0)}",
        "",
        "## Lineage",
        "- Crossref raw response -> parsed raw records -> cleaned dataframe -> Chroma index -> evaluation answers/metrics.",
        f"- Raw records path: {source_summary.get('raw_records_path', '')}",
        f"- Clean data path: {source_summary.get('clean_path', '')}",
        f"- Embeddings manifest: {source_summary.get('embeddings_path', '')}",
        "",
        "## Evaluation Metrics",
    ]
    for key in ["samples", "retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"]:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    if "ragas" in metrics:
        lines.append(f"- ragas: {metrics['ragas']}")

    lines.extend(
        [
            "",
            "## Data Quality",
            f"- Passed: {quality.get('passed')}",
            f"- Failed checks: {quality.get('failed_count', 0)} / {quality.get('total_checks', 0)}",
            "",
            "## Freshness",
            f"- Latest published: {freshness.get('latest_published')}",
            f"- Oldest published: {freshness.get('oldest_published')}",
            f"- Stale rows: {freshness.get('stale_rows')}",
            f"- Is fresh: {freshness.get('is_fresh')}",
            "",
        ]
    )
    write_text(report_path, "\n".join(lines))


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """Write a markdown comparison report for baseline, corrupted, and repaired runs."""
    metric_keys = ["samples", "retrieval_hit_rate", "mean_token_f1", "judge_accuracy", "mean_judge_score"]
    lines = [
        "# Corruption Impact Report",
        "",
        "## Metrics Comparison",
        "| Metric | Baseline | Corrupted | Repaired |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in metric_keys:
        lines.append(
            f"| {key} | {_fmt(baseline_metrics.get(key))} | {_fmt(corrupted_metrics.get(key))} | {_fmt(repaired_metrics.get(key))} |"
        )

    lines.extend(
        [
            "",
            "## Data Quality Comparison",
            "| Dataset | Passed | Failed checks | Total checks |",
            "| --- | ---: | ---: | ---: |",
            _quality_row("Corrupted", corrupted_quality),
            _quality_row("Repaired", repaired_quality),
            "",
            "## Freshness Comparison",
            "| Dataset | Latest published | Oldest published | Stale rows | Is fresh |",
            "| --- | --- | --- | ---: | ---: |",
            _freshness_row("Corrupted", corrupted_freshness),
            _freshness_row("Repaired", repaired_freshness),
            "",
            "## Interpretation",
            "- Data issue -> retrieval issue -> answer quality impact: corrupted data removes latest records, blanks summaries, injects noise, truncates titles, makes dates stale, and adds duplicate rows. These issues can reduce retrieval hit rate and answer overlap because the index contains incomplete or misleading context.",
            "- Repair recovery: the repaired dataset is rebuilt from the raw records and cleaned again, so quality gates and freshness should move back toward the baseline.",
            "- Traceability: answer artifacts contain question, ground truth, retrieved document IDs, retrieved contexts, retrieval hit, token F1, and judge verdict. Use those files to debug an answer back to the retrieved paper and source data.",
            "",
        ]
    )
    write_text(report_path, "\n".join(lines))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _quality_row(label: str, quality: dict[str, Any]) -> str:
    return (
        f"| {label} | {quality.get('passed')} | "
        f"{quality.get('failed_count', '')} | {quality.get('total_checks', '')} |"
    )


def _freshness_row(label: str, freshness: dict[str, Any]) -> str:
    return (
        f"| {label} | {freshness.get('latest_published')} | {freshness.get('oldest_published')} | "
        f"{freshness.get('stale_rows')} | {freshness.get('is_fresh')} |"
    )
