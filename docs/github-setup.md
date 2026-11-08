# GitHub upload and release

Repository name: `video-recsys-research-agent`

About description:
> Reproducible recommendation research with PyTorch, a genre Transformer, MovieLens evaluation, and a bounded LangGraph agent. Includes Recall@K, normalized entropy, ablations, MLflow tracking, and measured training costs.

Topics: `recommender-system`, `deep-learning`, `pytorch`, `transformers`, `pyspark`,
`mlflow`, `optuna`, `langgraph`, `machine-learning`, `research`, `movielens`.

Upload the extracted folder's contents into the repository root. Include
`.gitignore`, `.dockerignore` and `.github/workflows/`. If browser upload skips
the workflow folder, use Add file → Create new file, enter each full workflow
path, and paste that local file's exact contents. Do not upload data/ or runs/.

CI runs on push. Its synthetic tests do not download MovieLens or require
accepting dataset terms on a hosted runner. The Spark job validates a synthetic
fixture separately. A successful CI is not evidence that your own local training
run has finished; the bundled aggregate report documents the author's run.

After CI succeeds, Actions → Release → Run workflow publishes the version in
pyproject.toml as a source ZIP and Python wheel. It does not publish to PyPI or
GHCR, so an empty GitHub Packages panel is expected. A wheel release asset is
a downloadable Python package, not a GitHub registry package.

For later releases change the project version and release notes first. Do not
rerun the same version to replace existing release assets. GitHub Actions has
not been executed in your repository until you upload these files.

Optional: publish the aggregate-only reports/ directory with GitHub Pages.
No training data or checkpoint is needed for the static report. Use a real live
URL for Website only after deploying and checking it.
