# Phase 1 Baseline Report

## Source
- Source API: Crossref REST API
- Query: agentic retrieval augmented generation large language model
- Filter: from-pub-date:2025-12-12,has-abstract:true
- Raw records: 23
- Clean records: 23

## Lineage
- Crossref raw response -> parsed raw records -> cleaned dataframe -> Chroma index -> evaluation answers/metrics.
- Raw records path: D:\LAB\Day-10-Data-Pipeline-Data-Observability-main\data\raw\crossref_records.json
- Clean data path: D:\LAB\Day-10-Data-Pipeline-Data-Observability-main\data\clean\papers_clean.json
- Embeddings manifest: D:\LAB\Day-10-Data-Pipeline-Data-Observability-main\data\embeddings\papers_embeddings.json

## Evaluation Metrics
- samples: 32
- retrieval_hit_rate: 1.0
- mean_token_f1: 1.0
- judge_accuracy: 1.0
- mean_judge_score: 5
- ragas: {'skipped': 'Set RUN_RAGAS=1 to enable the slower Ragas pass.'}

## Data Quality
- Passed: True
- Failed checks: 0 / 13

## Freshness
- Latest published: 2026-06-02
- Oldest published: 2025-12-19
- Stale rows: 0
- Is fresh: True
