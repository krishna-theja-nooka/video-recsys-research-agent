# Three-minute research demonstration

1. Open `reports/index.html`. Explain that these are measured aggregate results
   from a movie-rating proxy dataset, not online video engagement.
2. Show the baseline, no-content ablation and popularity result. Discuss the
   different behavior of ranking recall and probability calibration (NE).
3. Open `reports/decision.json`. Show the three rejected proposals and the
   unchanged training hash. The agent kept the baseline rather than claiming improvement.
4. Open `reports/research-report.md`. Discuss cold-start exclusions, 52 evaluated
   positive test users, the confidence intervals and rejected-trial training cost.
5. Run a synthetic test or a local one-seed smoke experiment using README commands.
6. Open your local MLflow UI after training and compare validation/test metrics.

Do not present the optional Optuna study's best validation score as a test result.
Do not suggest the Transformer understands video content: it encodes genre tokens.
A good interview discussion includes why popularity remains competitive and what
additional data would be needed for a real video recommendation study.
