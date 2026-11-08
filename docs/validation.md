# Executed validation

The following checks were executed in a Linux Python 3.12 environment for this
project delivery (2026-11-08 Pacific time):

| Check | Observed result |
|---|---|
| Pytest including the opt-in Spark integration | 21 passed |
| Ruff lint and formatting | Passed |
| Dependency consistency (`pip check`) | No broken requirements |
| Python wheel build | Passed |
| Full MovieLens experiment | Nine training runs, three seeds, eight epochs |
| Repeated fixed experiment | Validation and test metrics exactly matched |
| Optuna exploratory study | Two completed validation-only trials |
| PySpark / Python canonical splits | Exact matching hashes on all 100K rows |
| Agent selection | All three interventions rejected; baseline retained |
| Model-source fingerprint | Matches the bundled `reclab/*.py` implementation |

The Spark integration ran with PySpark 3.5.5 and local OpenJDK 21.0.13;
JDK 17 is recommended and configured for the GitHub Spark job.
Two upstream MLflow/Pydantic deprecation warnings were observed; tests passed.

Windows and Docker were not executed in this build environment. Their setup
instructions and CI configuration are included, but remote GitHub CI cannot
run until these files are committed to the user's repository. No registry image
has been published. The tests are synthetic; only the reported research run
uses the downloaded public dataset.

`reports/reproduction-check.json` records repeated-run equality; the changed
second-run instrumentation adds the no-content paired bootstrap. No architecture,
training hyperparameter, candidate list, selection threshold or split was changed.
The source fingerprint is recorded in `reports/results.json`.

The benchmark uses only 52 positive test users after warm-start filtering. Its
large uncertainty and narrow coverage are central limitations, not details to omit.

The benchmark PNG was rendered and visually inspected. Automated browser layout
verification could not complete in this environment; no browser-test pass is claimed.
