# Local Laptop Runbook

This runbook describes the local recovery workflow for en-de mBART attention experiments on the RTX 4070 Laptop GPU machine.

## Current Local Baselines

- Smoke 12: 12 processed, 0 skipped, 56 token rows, 12 sentence rows, 2,304 sentence-head rows.
- 1k: 1,000 selected, 19 duplicate IDs rewritten, 1,000 processed, 45,099 token rows, 192,000 sentence-head rows.
- 10k: 10,000 selected, 474 duplicate IDs rewritten, 10,000 processed, 0 skipped, 450,461 token rows, 10,000 sentence rows, 1,920,000 sentence-head rows.
- 10k best sentence-head model: logistic regression, layer 6 head 13, test F1 0.711, test ROC-AUC 0.727.
- 10k token localization remained weak: best test F1 was about 0.137.

## 50k Local Run

Start the detached local run from the repository root:

```powershell
.\scripts\run_local_50k.ps1
```

Check status at any time:

```powershell
.venv\Scripts\python.exe -m lucid_decoders.tools.local_status `
  --run-id en_de_50k `
  --processed-dir data/processed/en_de_50k `
  --artifacts-dir artifacts/en_de_50k
```

Resume safely after a shutdown or interrupted run:

```powershell
.venv\Scripts\python.exe -m lucid_decoders.tools.local_run `
  --run-id en_de_50k `
  --processed-dir data/processed/en_de_50k `
  --artifacts-dir artifacts/en_de_50k `
  --normalized-source data/processed/en_de/all_trainable.jsonl `
  --device cuda `
  --chunk-size 250 `
  --seed 13 `
  --train-per-label 24037 `
  --validation-per-label 758 `
  --test-per-label 205 `
  --head-train-jobs 8 `
  --persist-head-models best
```

## Output Layout

- `data/processed/en_de_50k/all_trainable.jsonl`: sampled 50k subset.
- `data/processed/en_de_50k/chunks/`: resumable extraction chunk outputs.
- `data/processed/en_de_50k/*_features.parquet`: merged feature tables.
- `artifacts/en_de_50k/run_state.json`: durable stage state.
- `artifacts/en_de_50k/logs/`: heartbeat, run log, stage status, stdout/stderr, lock, and launcher PID files.
- `artifacts/en_de_50k/<model_type>/...`: classifier artifacts.
- `artifacts/en_de_50k/run_summary.json`: final run metrics and top-head report.

## Recovery Rules

- Extraction always uses chunk resume and retries missing or incomplete chunks.
- Token and sentence training stages are skipped only when `metrics.json`, `model.pkl`, and `test_predictions.parquet` are complete.
- Sentence-head training writes one recovery directory per `(layer_id, head_id)` and skips completed heads on resume.
- Sentence-head sweeps save all metrics and predictions, but only `best_model.pkl` by default.
- CUDA is required for the 50k extraction. If CUDA is unavailable, the run stops instead of falling back to CPU.

