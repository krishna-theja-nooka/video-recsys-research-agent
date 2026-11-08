# Preregistered reference protocol

This specification is fixed before the bundled real-data benchmark is run. Negative
results must not be removed. The report includes a source fingerprint and split digests.

- Data: MovieLens 100K, explicit ratings 1–5. Positive means rating >=4.
- Global chronological split: timestamp boundaries at sorted row indices 80,000 and
  90,000; ties remain together. Fit user/item maps on train. Exclude and count
  cold-start validation/test interactions. Do not use demographic attributes.
- Candidate catalog: all training items. Mask earlier observed items, regardless
  of their rating. Exact full-catalog ranking; no sampled negative evaluation.
- Model: user and item ID embeddings, dimension 32, dot product / sqrt(32),
  user/item/global biases. Item vector adds mean-pooled genre-token BERT output.
  BERT: one layer, four heads, feedforward width 64, zero dropout, randomly
  initialized. Genre IDs 1–19; padding 0; no pretrained model/tokenizer download.
- Objective: binary cross-entropy on observed ratings, unweighted for baseline.
  AdamW lr 0.003, weight decay 0.0001, gradient clipping 1.0, batch <=1,024,
  eight epochs, CPU two threads, deterministic PyTorch operations.
- Seeds: 42, 43, 44. Selection uses seed 42 validation only. Repeat baseline,
  selected intervention and no-content ablation on all three seeds.
- Three candidate weighting interventions: recency [0.8,1.2], tail upweight
  1.2, and rating-3 downweight 0.8. Every candidate starts from the same seed,
  initialization, raw training rows, optimizer settings and shuffle stream.
- Gate: validation Recall@10 at least baseline +0.001 and NE no greater than
  baseline ×1.02. Pick highest qualifying recall; otherwise keep baseline.
  Never update the baseline to a winning trial while searching.
- The selected policy is written to decision.json before the test partition
  is loaded. Rejected candidates are not scored on test. There is no
  production promotion, online intervention, or automatic code modification.
- Final evaluation: macro Recall@10 and NDCG@10 over eligible positive users;
  binary NE over all retained held-out ratings. Record denominators.
- Uncertainty: seed sample standard deviation plus 2,000 paired user-bootstrap
  resamples for selected-minus-baseline, separately by seed.
- Costs: model parameters, examples processed, optimizer steps, training wall
  and CPU seconds, full run wall time. Count rejected search trials in total cost.
- Optuna: optional separate exploratory validation-only study, max eight trials,
  seed 42, four epochs, dims {16,32}, lr [0.001,0.01]. Its results do not alter
  this benchmark; do not compare its best validation score with held-out test scores.

The single fixed test split is a local experimental convention, not a security
sandbox. A developer who can edit the code can access files. Repeatedly revising
this protocol after seeing test results would invalidate its confirmatory meaning.
