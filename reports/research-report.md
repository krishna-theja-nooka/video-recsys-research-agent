# Research report: bounded data interventions in movie recommendation

## Abstract

This experiment tests a two-tower recommender with a small genre Transformer and a deterministic LangGraph data-intervention controller. MovieLens 100K is a movie-rating proxy for video discovery, not a short-video impression benchmark. All figures below are measured locally; no online engagement or production impact is claimed.

## Protocol

Interactions are globally sorted by timestamp, user and item. Timestamp boundaries near 80% and 90% define train/validation/test. Ties remain in one partition. Mappings and eligible catalog are fitted on train only. Unseen users/items in later partitions are excluded and counted. Ratings >=4 are positive; ratings 1–3 are observed negative labels, not unobserved impressions. Validation selects only the data intervention. Test is opened after decision.json is written; failed proposals are never scored on test. All hyperparameters and epochs remain fixed.

Selected intervention: **baseline**. Seeds: [42, 43, 44]. Epochs per model: 8.

## Measured held-out results

| Variant | Recall@10 mean ± seed SD | NE mean ↓ | Train wall seconds mean | Seeds |
|---|---:|---:|---:|---:|
| baseline | 0.0870 ± 0.0094 | 0.8642 | 7.80 | 3 |
| no_content | 0.0727 ± 0.0088 | 0.8149 | 1.23 | 3 |
| popularity | 0.0894 | 0.8941 | no gradient training | deterministic |

Popularity ranks training positive counts; its pointwise prediction uses item positive rates smoothed by 20 global-prior pseudo-observations. The no_content ablation removes the genre encoder, leaving ID embeddings and biases. This also changes parameter count and compute, so it is not a capacity-matched control.

## Validation-only intervention trials

| Intervention | Recall@10 | NE | Train seconds |
|---|---:|---:|---:|
| baseline | 0.0471 | 0.9082 | 7.80 |
| recency | 0.0415 | 0.9109 | 7.58 |
| tail | 0.0465 | 0.9070 | 7.65 |
| ambiguous | 0.0355 | 0.9160 | 7.69 |
| no_content | 0.0667 | 0.8686 | 1.19 |

The controller tests exactly three allowlisted interventions: linear recency weights 0.8–1.2; weight 1.2 for items at or below median training item frequency; and weight 0.8 for rating-3 examples. No rows, labels, validation data or test data are edited. Approval requires validation Recall@K >= frozen baseline + 0.001 and normalized entropy <= baseline × 1.02. The best qualifying Recall@K wins, otherwise the baseline is retained. The intervention is chosen with the first seed only, then replicated with remaining paired seeds.

## Paired test uncertainty

These are percentile 95% paired user-bootstrap intervals, conditional on each trained model and split (2,000 resamples). If selected equals baseline, its zero interval is structural, not evidence of statistical equivalence. They are not across-dataset or causal confidence intervals. Seed variability is reported separately above.

| Comparison | Seed | Recall delta | Lower | Upper |
|---|---:|---:|---:|---:|
| selected_minus_baseline | 42 | 0.0000 | 0.0000 | 0.0000 |
| no_content_minus_baseline | 42 | -0.0324 | -0.0921 | 0.0174 |
| selected_minus_baseline | 43 | 0.0000 | 0.0000 | 0.0000 |
| no_content_minus_baseline | 43 | -0.0058 | -0.0689 | 0.0565 |
| selected_minus_baseline | 44 | 0.0000 | 0.0000 | 0.0000 |
| no_content_minus_baseline | 44 | -0.0048 | -0.0690 | 0.0561 |

## Training-cost accounting

Total measured gradient-training wall time: 50.02 seconds across 9 training runs, including rejected proposals. Full experiment wall time: 50.67 seconds including validation, tracking and final scoring (excluding report rendering and checkpoint serialization). results.json records optimizer steps, examples processed, parameter counts and CPU process time per run. CPU time can exceed wall time with multiple threads. There is no measured energy, GPU-hour, cloud-dollar or peak-memory claim. Optional Optuna tuning is separate and additional, with costs in tuning.json.

## Metrics

Recall@10 is macro-averaged over eligible users with at least one held-out positive: hits / all their held-out positives. Ranking uses the full training catalog, excluding earlier observed items (train for validation; train+validation for test). Lower ratings in the evaluation partition remain ranking candidates. NDCG uses binary relevance. Stable tie-breaking uses mapped item IDs.

Normalized entropy = mean binary cross-entropy / H(p), where p is the positive fraction of the evaluation partition. It is undefined for single-class labels; the code raises an error. NE=1 is the evaluation-prevalence constant predictor. This is rating-label NE, not CTR normalized entropy on a representative impression sample. No negative sampling is used; calibration cannot be generalized to clicks.

## Data coverage

| Partition | Original | Warm-start retained | Cold-start excluded |
|---|---:|---:|---:|
| train | 79999 | 79999 | 0 |
| valid | 10001 | 1520 | 8481 |
| test | 10000 | 1344 | 8656 |

## Limitations and interpretation

This small, historical, self-selected ratings dataset is not a randomized exposure log. Global temporal splitting excludes cold-start users and items, so accuracy describes only the retained warm-start population. Genre tokens are metadata, not video pixels or natural-language understanding. The Transformer is initialized randomly and trained from scratch; no pretrained model is downloaded. This is a deterministic experimental agent, not an LLM agent or autonomous deployment system. Multiple validation trials introduce selection bias. Three seeds are a small sample. A qualifying validation change may fail to improve test performance. Negative findings are retained. No result establishes Meta-scale performance, PhD-level novelty, or causality.

The benchmark keeps the baseline fixed. Optuna is an optional exploratory study that never accesses test or automatically changes the benchmark. A stronger study would preregister a new baseline, add capacity-matched encoders and multiple datasets, then validate on licensed video exposure/watch-time data.

## Reproduction

See README.md for pinned installation and commands. Raw data and checkpoints are excluded from distribution. Results include environment versions and split digests. CPU timings depend on hardware and load; equal seeds do not guarantee bitwise equality across different platforms.

## References

- Harper, F. M. and Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM TiiS 5(4), Article 19. https://doi.org/10.1145/2827872
- https://grouplens.org/datasets/movielens/100k/
- https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
- https://huggingface.co/docs/transformers/model_doc/bert
- https://docs.langchain.com/oss/python/langgraph/overview
