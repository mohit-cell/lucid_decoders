# Recovery Changelog

## Local Laptop Recovery

- Added atomic JSON, pickle, and table writes for classifier outputs.
- Added resumable sentence-head training with per-head recovery artifacts.
- Changed sentence-head model persistence default to `best`, avoiding large all-head random-forest pickles.
- Added process-backed sentence-head parallelism through `--n-jobs`.
- Added `lucid-run-local` to manage local run state, heartbeat logs, stage logs, locks, and resume behavior.
- Added `lucid-local-status` to inspect run state and print the safe resume command.
- Added `scripts/run_local_50k.ps1` to launch the 50k run outside the chat session.
- Added tracked local runbook documentation while keeping detailed run logs under ignored `artifacts/`.

