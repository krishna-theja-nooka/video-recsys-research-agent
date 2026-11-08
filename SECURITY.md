# Security and data handling

This is a local research application, not a hardened multi-tenant service.
The agent executes only prewritten train-data weighting functions. It has no
credentials, shell tool, arbitrary SQL or unrestricted code execution interface.
Training and evaluation budgets are bounded in configuration. A cryptographic
hash detects accidental split modification but is not an access-control system.

Keep data/, runs/, mlruns/, model checkpoints and virtual environments out of Git.
Use only aggregate reports in public releases. Never load untrusted PyTorch
checkpoints. The CLI saves local model state for research but does not offer an
untrusted checkpoint loading endpoint.

Run MLflow and the static report server on 127.0.0.1. Do not expose them publicly
without authentication, storage isolation, dependency review and operational controls.
Pinned packages make reproduction easier; they do not imply absence of vulnerabilities.
CI tests correctness and formatting, not a complete security audit.

Report vulnerabilities privately through GitHub's security reporting feature when
enabled. Do not include raw dataset records or tokens in a public issue.
