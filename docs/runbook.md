# Local research runbook

- Dataset permission: read the original GroupLens terms before passing
  `--accept-license`. Do not redistribute the downloaded archive or extracted rows.
- Download failure: confirm internet access to files.grouplens.org. Delete only
  the incomplete `ml-100k.tmp` if necessary and retry. A SHA mismatch is a stop
  condition; investigate before accepting a different archive.
- Out of memory or slow CPU: start with one seed and two epochs in a fresh
  output directory. Label those results as a smoke run, not the eight-epoch benchmark.
- Existing output: use a new `--output` path. Experiments refuse to overwrite
  completed or partial output directories. Preserve failures for investigation.
- Failed checksum: restore/reprepare the original data; do not edit the recorded
  checksum to disguise a changed split. Start a new experiment directory.
- PySpark on Windows: use JDK 17 and a Linux environment such as WSL for the
  optional Spark path if Windows Hadoop filesystem utilities cause errors.
  Python ingestion and CPU training do not need Java or Hadoop.
- MLflow: bind only to 127.0.0.1 and point to your experiment's mlruns directory.
  The packaged results viewer works without starting any service.
- Failed candidate: baseline retention is normal. Inspect decision.json and the
  validation metrics. Never loosen the gate after viewing test scores.
- Reproduction drift: compare pinned environment versions, input SHA256,
  split digests, model config and random seeds before comparing metrics.
