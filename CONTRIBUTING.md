# Contributing

Use Python 3.12, install the pinned environment and run the documented test commands.
Keep metric definitions and tests together. Preserve all failed experiments.
Propose protocol changes in a new experiment directory, with a new protocol version.
Do not choose changes using test results and then claim the same test is untouched.

Pull requests must include the reason for changes, test results and whether they
change data eligibility, metric denominators, model capacity or training budget.
Never commit downloaded data, trained model weights, MLflow local stores or credentials.
The tests use generated synthetic fixtures and need no external dataset access.
