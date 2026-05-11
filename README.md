# lucid_decoders

Attention-based hallucination detection for English-to-German machine translation.

The pipeline normalizes WMT22/WMT23 en-de quality-estimation data, runs mBART with
teacher-forced target translations and `output_attentions=True`, converts attention
tensors into compact features, and trains:

- a token-level hallucination/error localization classifier
- a sentence-level hallucination probability classifier
- one sentence-level classifier per decoder `(layer, head)` for attention-head ranking

## Setup

For a fresh clone on macOS or Linux:

```bash
git clone https://github.com/mohit-cell/lucid_decoders.git
cd lucid_decoders
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

For a fresh clone on Windows PowerShell:

```powershell
git clone https://github.com/mohit-cell/lucid_decoders.git
cd lucid_decoders
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The raw data submodules point to:

- `data/raw/wmt22`: `https://github.com/WMT-QE-Task/wmt-qe-2022-data.git`
- `data/raw/wmt23`: `https://github.com/WMT-QE-Task/wmt-qe-2023-data.git`

Validate the raw checkout before preprocessing:

```bash
lucid-validate-ende-data \
  --wmt22-root data/raw/wmt22 \
  --wmt23-root data/raw/wmt23
```

## One-Command Pipeline

Run the full pipeline:

```bash
lucid-run-ende-pipeline \
  --stage all \
  --wmt22-root data/raw/wmt22 \
  --wmt23-root data/raw/wmt23 \
  --processed-dir data/processed/en_de \
  --artifacts-dir artifacts/en_de \
  --device cuda
```

Use `--device cpu` if GPU is unavailable. mBART extraction is the expensive stage.

Run one stage at a time with `--stage prepare`, `--stage extract`,
`--stage train-token`, `--stage train-sentence`, or `--stage train-heads`.

## Colab Runtime Recovery

If Colab disconnects during full-data extraction, reconnect, remount Drive, reinstall
the package, then inspect saved chunks:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content/drive/MyDrive/NLP/NLP_Project/lucid_decoders
!pip install -e .
!lucid-colab-recovery
```

The recovery command prints how many chunk reports were saved under
`data/processed/en_de_full_features/chunks`, whether merged Parquet outputs exist,
and the exact `extract-chunked` command to rerun. Rerunning that command is safe:
completed chunks are skipped because the pipeline uses resumable chunk reports.

## Manual Stage Commands

Prepare normalized en-de JSONL files:

```bash
lucid-prepare-ende \
  --wmt22-root data/raw/wmt22 \
  --wmt23-root data/raw/wmt23 \
  --output-dir data/processed/en_de
```

This writes both per-source files and canonical combined files:

- `all_examples.jsonl`: all exported examples, including audit-only rows
- `all_trainable.jsonl`: examples with source, translation, label, and split

WMT23 hallucination gold rows currently lack source sentences in the raw repo; they are
kept in `all_examples.jsonl` and excluded from `all_trainable.jsonl`.

Extract mBART attention features:

```bash
lucid-extract-mbart \
  --input data/processed/en_de/all_trainable.jsonl \
  --token-output data/processed/en_de/token_features.parquet \
  --sentence-output data/processed/en_de/sentence_features.parquet \
  --sentence-head-output data/processed/en_de/sentence_head_features.parquet \
  --model-name facebook/mbart-large-50-many-to-many-mmt \
  --source-lang en_XX \
  --target-lang de_DE \
  --device cuda \
  --require-sentence-label \
  --report-output data/processed/en_de/mbart_extraction_report.json
```

Train classifiers:

```bash
lucid-train-token \
  --features data/processed/en_de/token_features.parquet \
  --artifacts-dir artifacts/en_de/token_classifier

lucid-train-sentence \
  --features data/processed/en_de/sentence_features.parquet \
  --artifacts-dir artifacts/en_de/sentence_classifier

lucid-train-sentence-head \
  --features data/processed/en_de/sentence_head_features.parquet \
  --artifacts-dir artifacts/en_de/sentence_head_classifier
```

## Output Tree

```text
data/processed/en_de/
  all_examples.jsonl
  all_trainable.jsonl
  token_features.parquet
  sentence_features.parquet
  sentence_head_features.parquet
  mbart_extraction_report.json

artifacts/en_de/
  token_classifier/
    model.pkl
    metrics.json
    test_predictions.parquet
  sentence_classifier/
    model.pkl
    metrics.json
    test_predictions.parquet
  sentence_head_classifier/
    models_by_head.pkl
    metrics.json
    head_metrics.csv
    test_predictions.parquet
```

`head_metrics.csv` is sorted by validation performance and identifies the decoder
attention heads most predictive of sentence-level hallucination/error labels.

## Feature Contract

Token rows contain pooled cross-attention and decoder self-attention statistics across
layers and heads:

- entropy, max, variance, and top-k mass
- mean, std, min, and max summaries
- `self_to_cross_max_ratio`
- `self_to_cross_entropy_ratio`
- token position and source/target lengths

Sentence rows aggregate token features across target tokens with mean, max, min, and
std. Sentence-head rows aggregate each `(layer_id, head_id)` across target tokens and
include per-token self-to-cross ratio means.

## Project Code And Demo Checklist

This section is written for the final CSE 538 code/demo. The
source code is hosted at:

- GitHub packaged-code link: <https://github.com/mohit-cell/lucid_decoders.git>
- Original/current project codebase: this repository. The project was developed
  for the Lucid Decoders course project rather than forked from an external
  application codebase.
- WMT22 quality-estimation data: <https://github.com/WMT-QE-Task/wmt-qe-2022-data.git>
- WMT23 quality-estimation/hallucination data: <https://github.com/WMT-QE-Task/wmt-qe-2023-data.git>
- mBART model used for feature extraction: <https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt>

### Modified and Added Files

The table below lists the main project files and the concrete functions, classes,
or command entry points added or modified for the project. Test files are listed
by behavior because their individual `test_*` functions are intentionally narrow.

| File | Main functions, classes, or content added/modified |
| --- | --- |
| `README.md` | Setup instructions, pipeline commands, feature contract, and this course code/demo checklist. |
| `pyproject.toml` | Package metadata, runtime/dev dependencies, and console scripts: `lucid-preprocess`, `lucid-prepare-ende`, `lucid-sample-ende`, `lucid-validate-ende-data`, `lucid-extract-mbart`, `lucid-extract-mbart-chunked`, `lucid-run-ende-pipeline`, `lucid-train-token`, `lucid-train-sentence`, `lucid-train-sentence-head`, `lucid-eval-token`, `lucid-eval-sentence`, `lucid-plot-attention`, `lucid-colab-recovery`, `lucid-run-local`, and `lucid-local-status`. |
| `.gitmodules` | WMT22 and WMT23 raw-data submodule locations. |
| `.gitignore` | Ignores local virtual environments, generated processed data, and generated artifacts. |
| `scripts/run_local_50k.ps1` | PowerShell launcher for detached local 50k runs, PID files, stdout/stderr logs, stale active-run checks, and safe reruns. |
| `docs/local_laptop_runbook.md` | Tracked runbook for laptop setup, local run commands, recovery behavior, and artifact inspection. |
| `docs/recovery_changelog.md` | Tracked changelog describing recovery and resumability implementation. |
| `docs/project_history_and_results.md` | Full technical history of the smoke, 1k, 10k, and 50k runs, including implementation notes and metrics. |
| `docs/prompts_used.txt` | States that the runnable system uses no prompt-based inference and summarizes project-assistance prompts for transparency. |
| `src/lucid_decoders/data/preprocess.py` | `build_arg_parser`, `main`, and generic JSON/CSV/TSV normalization helpers for source, target, sentence-label, token-label, span, ID, and split fields. |
| `src/lucid_decoders/data/wmt.py` | WMT-oriented normalization helpers and CLI wrapper for raw JSONL/JSON/CSV/TSV inputs. |
| `src/lucid_decoders/data/prepare_ende.py` | `build_arg_parser`, `main`, `prepare_ende_datasets`, `is_trainable_for_attention`, `build_wmt22_sentence_examples`, `build_wmt22_word_examples`, `build_wmt23_task2_examples`, `build_wmt23_hallucination_gold_examples`, archive readers, span parsing/merging helpers, tag normalization, EOS stripping, and stable ID generation. |
| `src/lucid_decoders/data/sample_ende.py` | `build_arg_parser`, `main`, `sample_balanced_examples`, `has_token_supervision`, `make_example_ids_unique`, and split-order helpers for balanced smoke/1k/10k/50k subsets. |
| `src/lucid_decoders/data/validate_ende.py` | `RequiredPath`, `DataValidationIssue`, `DataValidationReport`, `validate_wmt_roots`, `raise_for_missing_data`, `build_arg_parser`, and `main` for raw data readiness checks. |
| `src/lucid_decoders/features/token_features.py` | Token-level attention summary feature construction and token row schema support. |
| `src/lucid_decoders/features/sentence_features.py` | `build_sentence_feature_frame` for sentence-level aggregation of token attention features. |
| `src/lucid_decoders/features/sentence_head_features.py` | `build_sentence_head_feature_rows`, `_collect_token_metrics`, `_summarize_metric_values`, and `_mean_ratio` for one row per sentence/layer/head. |
| `src/lucid_decoders/features/contracts.py` | `attention_summary_columns`, `sentence_aggregate_columns`, and `validate_feature_frame` feature-schema checks. |
| `src/lucid_decoders/io.py` | `ensure_parent_dir`, `read_jsonl`, `write_jsonl`, `write_jsonl_atomic`, `read_table`, `write_table`, `write_table_atomic`, `write_text_atomic`, `write_json_atomic`, and `temporary_path` for consistent file IO and atomic writes. |
| `src/lucid_decoders/ml.py` | `require_sklearn`, `get_default_feature_columns`, `build_estimator`, `predict_positive_proba`, `tune_threshold`, `binary_classification_metrics`, `empty_binary_classification_metrics`, `validate_training_frame`, `save_pickle`, `load_pickle`, and `save_json`. |
| `src/lucid_decoders/models/mbart_attention.py` | `MBartAttentionExtractor` and its loading/device/language/tokenization/alignment/extraction helpers, plus `load_examples`, `validate_example_for_extraction`, `build_arg_parser`, and `main`. |
| `src/lucid_decoders/models/mbart_attention_chunked.py` | `ChunkPaths`, `ChunkReport`, `build_arg_parser`, `main`, `run_chunked_extraction`, `process_chunk`, `merge_chunk_tables`, `build_final_report`, `build_chunk_paths`, `is_completed_chunk`, `read_chunk_report`, `atomic_write_table`, and `atomic_write_json`. |
| `src/lucid_decoders/pipeline.py` | `build_arg_parser`, `main`, `resolve_stages`, `run_prepare`, `run_extract`, `run_extract_chunked`, `run_train_token`, `run_train_sentence`, `run_train_heads`, `run_python`, and `print_stage`. |
| `src/lucid_decoders/tools/colab_recovery.py` | `build_arg_parser`, `main`, `collect_status`, chunk/report readers, file status helpers, Drive-backed path detection, resume-command generation, and status formatting. |
| `src/lucid_decoders/tools/local_run.py` | `build_arg_parser`, `main`, `run_local`, `run_stage`, `run_logged_command`, `run_environment_check`, command builders, stage-completion checks, report collection, classifier summaries, state/load/save helpers, lock handling, process checks, heartbeat writing, stage-status writing, run-log writing, and timestamp helpers. |
| `src/lucid_decoders/tools/local_status.py` | `build_arg_parser`, `main`, `collect_status`, `chunk_status`, `output_status`, `classifier_status`, `file_status`, `sum_int`, `build_resume_command`, and `format_status`. |
| `src/lucid_decoders/train/train_token_classifier.py` | `build_arg_parser` and `main` for token-level classifier training, threshold tuning, metric writing, model persistence, and prediction export. |
| `src/lucid_decoders/train/train_sentence_classifier.py` | `build_arg_parser` and `main` for sentence-level classifier training, threshold tuning, metric writing, model persistence, and prediction export. |
| `src/lucid_decoders/train/train_sentence_head_classifier.py` | `build_arg_parser`, `main`, `train_or_resume_head`, `build_head_dir`, `read_completed_head_result`, `cleanup_temp_files`, `fit_head_model`, `train_one_head`, `prefix_metrics`, `rank_score`, and `none_if_nan` for resumable per-head classifier training. |
| `src/lucid_decoders/eval/evaluate_token.py` | Token prediction evaluation entry point. |
| `src/lucid_decoders/eval/evaluate_sentence.py` | Sentence prediction evaluation entry point. |
| `src/lucid_decoders/analysis/plot_attention.py` | Attention plotting entry point for inspecting extracted attention behavior. |
| `tests/test_features.py` | Feature-contract and aggregation tests. |
| `tests/test_prepare_ende.py` | WMT en-de preparation and normalization tests. |
| `tests/test_sample_ende.py` | Balanced sampling, supervision filtering, and duplicate-ID rewrite tests. |
| `tests/test_mbart_attention_chunked.py` | Chunk manifest, resume, retry, and merge orchestration tests using fake extraction. |
| `tests/test_pipeline.py` | Pipeline stage resolution and command-construction regression tests. |
| `tests/test_training_smoke.py` | Small classifier-training smoke tests. |
| `tests/test_colab_recovery.py` | Recovery/status command generation tests for chunked extraction. |
| `tests/test_local_run_recovery.py` | Local run-state, heartbeat, stale-lock, and resume-command tests. |
| `tests/test_sentence_head_recovery.py` | Per-head resume, incomplete-head retry, merged metrics, predictions, and `persist-head-models=best` tests. |

### Training, Testing, and Demo Commands

Use the Setup section above to clone the repository, initialize submodules, create
a virtual environment, install the package, and run tests. The commands below are
PowerShell examples for validation, local demo runs, and artifact inspection.

Validate and prepare the raw WMT data:

```powershell
lucid-validate-ende-data `
  --wmt22-root data/raw/wmt22 `
  --wmt23-root data/raw/wmt23

lucid-prepare-ende `
  --wmt22-root data/raw/wmt22 `
  --wmt23-root data/raw/wmt23 `
  --output-dir data/processed/en_de
```

Run a small real-mBART smoke/demo experiment. This creates 12 examples, extracts
features in small chunks, and trains the logistic-regression versions of the three
classifiers:

```powershell
lucid-run-local `
  --run-id en_de_demo_12 `
  --processed-dir data/processed/en_de_demo_12 `
  --artifacts-dir artifacts/en_de_demo_12 `
  --normalized-source data/processed/en_de/all_trainable.jsonl `
  --device cuda `
  --chunk-size 4 `
  --train-per-label 2 `
  --validation-per-label 2 `
  --test-per-label 2 `
  --model-types logistic_regression `
  --head-train-jobs 1 `
  --persist-head-models best
```

Run the completed 50k local laptop experiment through the detached launcher:

```powershell
.\scripts\run_local_50k.ps1
```

Check status or recover a stopped run:

```powershell
lucid-local-status `
  --run-id en_de_50k `
  --processed-dir data/processed/en_de_50k `
  --artifacts-dir artifacts/en_de_50k
```

The status command prints the exact safe resume command. The underlying recovery
runner can also be called directly:

```powershell
lucid-run-local `
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

Manual 50k sampling and chunked extraction, equivalent to the sampler/extractor
stages inside `lucid-run-local`:

```powershell
lucid-sample-ende `
  --input data/processed/en_de/all_trainable.jsonl `
  --output data/processed/en_de_50k/all_trainable.jsonl `
  --train-per-label 24037 `
  --validation-per-label 758 `
  --test-per-label 205 `
  --seed 13 `
  --summary-output data/processed/en_de_50k/sample_summary.json

lucid-extract-mbart-chunked `
  --input data/processed/en_de_50k/all_trainable.jsonl `
  --token-output data/processed/en_de_50k/token_features.parquet `
  --sentence-output data/processed/en_de_50k/sentence_features.parquet `
  --sentence-head-output data/processed/en_de_50k/sentence_head_features.parquet `
  --report-output data/processed/en_de_50k/mbart_extraction_report.json `
  --chunks-dir data/processed/en_de_50k/chunks `
  --chunk-size 250 `
  --model-name facebook/mbart-large-50-many-to-many-mmt `
  --source-lang en_XX `
  --target-lang de_DE `
  --device cuda `
  --resume
```

Manual baseline/system training commands. Logistic regression is the simplest
baseline; random forest and MLP are the comparison systems:

```powershell
$ModelType = "logistic_regression"  # or random_forest, mlp

lucid-train-token `
  --features data/processed/en_de_50k/token_features.parquet `
  --artifacts-dir artifacts/en_de_50k/$ModelType/token_classifier `
  --model-type $ModelType

lucid-train-sentence `
  --features data/processed/en_de_50k/sentence_features.parquet `
  --artifacts-dir artifacts/en_de_50k/$ModelType/sentence_classifier `
  --model-type $ModelType

lucid-train-sentence-head `
  --features data/processed/en_de_50k/sentence_head_features.parquet `
  --artifacts-dir artifacts/en_de_50k/$ModelType/sentence_head_classifier `
  --model-type $ModelType `
  --min-train-examples 20 `
  --n-jobs 8 `
  --resume `
  --persist-head-models best
```

Inspect generated reports and metrics:

```powershell
Get-Content artifacts/en_de_50k/run_summary.json
Get-Content data/processed/en_de_50k/mbart_extraction_report.json
Import-Csv artifacts/en_de_50k/logistic_regression/sentence_head_classifier/head_metrics.csv |
  Select-Object -First 10
```

### Trained Models and Training Data

Generated data and trained artifacts are intentionally ignored by git because the
50k processed features and random-forest models are large. They are reproducible
from the commands above and from the linked WMT/mBART sources.

Local trained model locations from the completed 50k run:

- Logistic regression:
  - `artifacts/en_de_50k/logistic_regression/token_classifier/model.pkl`
  - `artifacts/en_de_50k/logistic_regression/sentence_classifier/model.pkl`
  - `artifacts/en_de_50k/logistic_regression/sentence_head_classifier/best_model.pkl`
- MLP:
  - `artifacts/en_de_50k/mlp/token_classifier/model.pkl`
  - `artifacts/en_de_50k/mlp/sentence_classifier/model.pkl`
  - `artifacts/en_de_50k/mlp/sentence_head_classifier/best_model.pkl`
- Random forest:
  - `artifacts/en_de_50k/random_forest/token_classifier/model.pkl`
  - `artifacts/en_de_50k/random_forest/sentence_classifier/model.pkl`
  - `artifacts/en_de_50k/random_forest/sentence_head_classifier/best_model.pkl`

The random-forest token model is approximately 9.5 GB locally, and the full
`artifacts/en_de_50k/random_forest` tree is about 10 GB. It is not suitable for
GitHub hosting. The report and README therefore distinguish the GitHub code
package from local/generated trained artifacts.

Training data and extracted features from the completed 50k run:

- Normalized source subset: `data/processed/en_de_50k/all_trainable.jsonl`
- Sampling summary: `data/processed/en_de_50k/sample_summary.json`
- Token features: `data/processed/en_de_50k/token_features.parquet`
- Sentence features: `data/processed/en_de_50k/sentence_features.parquet`
- Sentence-head features: `data/processed/en_de_50k/sentence_head_features.parquet`
- Extraction report: `data/processed/en_de_50k/mbart_extraction_report.json`
- Original raw data: WMT22 and WMT23 repositories linked above.
- Feature extractor model: Hugging Face mBART model linked above.

### Prompts

The runnable hallucination-detection system does not use prompt-based inference.
It is a deterministic data processing, mBART feature extraction, and supervised
classifier-training pipeline. `docs/prompts_used.txt` documents this explicitly
and summarizes project-assistance prompts for transparency.

### Observed 50k Run Environment

The completed 50k local run used the following observed environment. These are
not the minimum required versions; package minimums are defined in
`pyproject.toml`.

| Component | Version or value |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA used by PyTorch | 12.8 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| transformers | 5.7.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |
| numpy | 2.4.4 |
| datasets | 4.8.5 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| tqdm | 4.67.3 |
| pytest | Installed through `.[dev]` |

CUDA is strongly recommended for real mBART extraction. The 50k run requires CUDA
for practical runtime; the small 12-example smoke run can be used as the TA demo
if a shorter end-to-end execution is needed.
