# Corruption Impact Report

## Metrics Comparison
| Metric | Baseline | Corrupted | Repaired |
| --- | ---: | ---: | ---: |
| samples | 32 | 32 | 32 |
| retrieval_hit_rate | 1.0000 | 0.5312 | 1.0000 |
| mean_token_f1 | 1.0000 | 0.4954 | 1.0000 |
| judge_accuracy | 1.0000 | 0.4688 | 1.0000 |
| mean_judge_score | 5 | 2.8750 | 5 |

## Data Quality Comparison
| Dataset | Passed | Failed checks | Total checks |
| --- | ---: | ---: | ---: |
| Corrupted | False | 4 | 13 |
| Repaired | True | 0 | 13 |

## Freshness Comparison
| Dataset | Latest published | Oldest published | Stale rows | Is fresh |
| --- | --- | --- | ---: | ---: |
| Corrupted | 2026-05-31 | 2018-01-01 | 4 | False |
| Repaired | 2026-06-02 | 2025-12-19 | 0 | True |

## Interpretation
- Data issue -> retrieval issue -> answer quality impact: corrupted data removes latest records, blanks summaries, injects noise, truncates titles, makes dates stale, and adds duplicate rows. These issues can reduce retrieval hit rate and answer overlap because the index contains incomplete or misleading context.
- Repair recovery: the repaired dataset is rebuilt from the raw records and cleaned again, so quality gates and freshness should move back toward the baseline.
- Traceability: answer artifacts contain question, ground truth, retrieved document IDs, retrieved contexts, retrieval hit, token F1, and judge verdict. Use those files to debug an answer back to the retrieved paper and source data.
