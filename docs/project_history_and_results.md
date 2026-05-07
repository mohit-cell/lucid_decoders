# Lucid Decoders Project History And Results

Last updated: 2026-05-05

This document is the consolidated technical record for the en-de hallucination
detection work on this branch. It covers the original pipeline goal, the
implementation work completed, all local runs from the 12-example smoke test
through the completed 50k laptop run, the classifier outputs, the recovery
infrastructure, and the current project status.

The run artifacts remain the source of truth. This document pulls from the
local JSONL, parquet, JSON, CSV, and log outputs already present under
`data/processed/` and `artifacts/`. It does not describe a new experiment and
does not rely on rerunning mBART or any classifier.

## 1. Executive Summary

The branch now contains a working attention-feature pipeline for English to
German hallucination/error detection using real `facebook/mbart-large-50-many-to-many-mmt`
attention outputs. The implemented pipeline starts from normalized WMT22/WMT23
examples, samples controlled en-de subsets, extracts teacher-forced mBART
cross-attention and decoder self-attention features, trains token-level,
sentence-level, and sentence-level layer/head classifiers, and produces ranked
attention-head results.

The pipeline is no longer just a prototype. It has been scaled through four
local runs:

| run | role | examples | extraction status | model coverage |
|---|---:|---:|---|---|
| `en_de_smoke_12` | correctness smoke | 12 | complete, non-chunked | logistic regression token, sentence, heads |
| `en_de_1k` | first practical baseline | 1,000 | complete, non-chunked | logistic regression token, sentence, heads |
| `en_de_10k` | first reliable model-comparison run | 10,000 | complete, chunked, 40 chunks | logistic regression, random forest, MLP |
| `en_de_50k` | current main local result | 50,000 | complete, chunked, 200 chunks | logistic regression, random forest, MLP |

The strongest current result is not token localization. The strongest and most
stable evidence is at the sentence and sentence-head levels:

| current best area | best 50k result | interpretation |
|---|---|---|
| Sentence classifier | logistic regression, test F1 `0.712215`, test ROC-AUC `0.695324` | Whole-sentence hallucination/error prediction is working better than token localization. |
| Sentence-head classifier | logistic regression, layer 6 head 13, test F1 `0.708171`, test ROC-AUC `0.728447` | Individual mBART heads contain predictive sentence-level signal. |
| Token classifier | MLP, test F1 `0.149076`, test ROC-AUC `0.596523` | Token localization remains weak, though the 50k MLP improved over the 10k MLP. |

The 50k run completed end to end on the local Windows laptop with CUDA enabled
on the NVIDIA GeForce RTX 4070 Laptop GPU. It produced:

| output | value |
|---|---:|
| sampled JSONL rows | 50,000 |
| processed examples | 50,000 |
| skipped examples | 0 |
| extraction chunks | 200/200 |
| token feature rows | 2,294,505 |
| sentence feature rows | 50,000 |
| sentence-head feature rows | 9,600,000 |
| test token prediction rows per model | 15,690 |
| test sentence prediction rows per model | 410 |
| test sentence-head prediction rows per model | 78,720 |

The project is working as an end-to-end scalable experiment pipeline. The next
engineering priority is to preserve and commit the branch state. The next
scientific priority is to improve token localization diagnostics before scaling
far beyond 50k.

## 2. Original Pipeline Goal

The original target pipeline was:

1. Start with WMT22/WMT23 en-de data.
2. Normalize raw records into JSONL examples with:
   - source sentence
   - generated German translation
   - sentence-level hallucination/error label
   - token/span labels where available
   - train/validation/test split
3. Run mBART with `output_attentions=True` on each source and provided
   translation.
4. Use teacher forcing. The model attends while reading the dataset translation,
   not while generating a new translation.
5. Extract compact statistics from:
   - decoder cross-attention from target tokens to source tokens
   - decoder self-attention over target-side prefix/context
6. Train:
   - a token-level classifier for localization
   - a sentence-level classifier for final sentence probability
   - a sentence-level `(layer, head)` classifier for attention-head ranking
7. Produce:
   - token-level hallucination/error predictions
   - sentence-level hallucination/error predictions
   - ranked mBART attention heads

The implementation now covers all of these core goals.

## 3. Implementation Work Completed

### Dataset Preparation And Sampling

The normalized full en-de dataset exists under `data/processed/en_de/`.

Key full processed files:

| file | size bytes | role |
|---|---:|---|
| `data/processed/en_de/all_examples.jsonl` | 136,322,442 | all normalized examples |
| `data/processed/en_de/all_trainable.jsonl` | 134,979,617 | trainable normalized examples |
| `data/processed/en_de/wmt22_ende_sentence_mqm.jsonl` | 24,643,992 | WMT22 sentence MQM-derived data |
| `data/processed/en_de/wmt22_ende_word_mqm.jsonl` | 81,619,960 | WMT22 word/token MQM-derived data |
| `data/processed/en_de/wmt23_ende_hallucination_gold.jsonl` | 1,342,825 | WMT23 hallucination gold data |
| `data/processed/en_de/wmt23_ende_task2.jsonl` | 28,715,665 | WMT23 task data |

The sampler now supports controlled per-split/per-label sampling, positive
token/span supervision requirements, fixed seed reproducibility, summary output,
and duplicate `example_id` rewriting. Duplicate rewrite is important because the
raw trainable pool can contain logically duplicated IDs when examples are
merged from multiple WMT-derived sources. The sampler rewrites duplicates in
the sampled subset so downstream parquet joins and prediction tables preserve a
unique example identity.

The sampler was used with seed `13` for 1k, 10k, and 50k runs. Positive examples
were required to have token/span supervision, so token-level training had
positive labeled rows.

### mBART Attention Extraction

The base extractor runs `facebook/mbart-large-50-many-to-many-mmt` with:

| setting | value |
|---|---|
| source language | `en_XX` |
| target language | `de_DE` |
| source max length | 256 |
| target max length | 256 |
| top-k for attention mass | 3 |
| teacher forcing | yes |
| attentions used | decoder cross-attentions and decoder self-attentions |

The non-chunked extractor was used for the 12-example and 1k runs. The chunked
extractor was added for 10k and 50k.

### Chunked And Resumable Extraction

The new CLI `lucid-extract-mbart-chunked` processes examples in stable chunks.
For the 10k and 50k runs, the chunk size was `250`.

The chunked extractor:

- loads mBART once per run
- processes a chunk of examples
- writes per-chunk token, sentence, sentence-head parquet files
- writes a per-chunk JSON report
- skips already completed chunks when `--resume` is used
- merges all chunk outputs into final feature parquet files
- writes `mbart_extraction_report.json`
- uses temp-to-final writes for chunk outputs so interrupted chunks are retried
  instead of trusted

The final merged outputs are:

| output | meaning |
|---|---|
| `token_features.parquet` | one row per generated target token |
| `sentence_features.parquet` | one row per full translated sentence |
| `sentence_head_features.parquet` | one row per sentence per decoder `(layer_id, head_id)` |
| `mbart_extraction_report.json` | processed/skipped counts, row counts, chunk metadata |

### Pipeline Runner

The pipeline runner now supports:

| stage | behavior |
|---|---|
| `prepare` | prepares normalized data from raw WMT roots |
| `extract` | non-chunked mBART attention extraction |
| `extract-chunked` | resumable chunked mBART attention extraction |
| `train-token` | trains token classifier |
| `train-sentence` | trains sentence classifier |
| `train-heads` | trains one sentence classifier per decoder layer/head |

The runner forwards model type, seed, threshold, head train jobs,
`min-train-examples`, and head model persistence controls.

### Classifier Training

Three model types are supported:

| model type | role in experiments |
|---|---|
| `logistic_regression` | stable linear baseline and easiest model to interpret |
| `random_forest` | nonlinear tree baseline |
| `mlp` | nonlinear neural baseline using existing scikit-learn dependencies |

The token and sentence classifiers write:

- `metrics.json`
- `model.pkl`
- `test_predictions.parquet`

The sentence-head classifier writes:

- `metrics.json`
- `head_metrics.csv`
- `test_predictions.parquet`
- `best_model.pkl` when `--persist-head-models best`
- `best_model_info.json`
- per-head recovery work directories under `head_work/`

### Atomic Output Writes

The IO layer now includes atomic writes for:

- JSONL
- JSON
- text
- parquet/CSV/TSV/JSONL tables
- pickle model artifacts

This reduces the chance that an interrupted run leaves a file that appears
complete but contains partial content.

### Local Laptop Recovery

The new local orchestrator CLI is `lucid-run-local`. It manages one local run by
`run_id`, writes durable state, emits logs, and skips completed stages on resume.

The new status CLI is `lucid-local-status`. It reads run state, lock files,
chunk reports, model outputs, and logs, then prints the safe resume command.

The 50k run used:

| config key | value |
|---|---|
| run id | `en_de_50k` |
| processed dir | `data/processed/en_de_50k` |
| artifacts dir | `artifacts/en_de_50k` |
| normalized source | `data/processed/en_de/all_trainable.jsonl` |
| model | `facebook/mbart-large-50-many-to-many-mmt` |
| source lang | `en_XX` |
| target lang | `de_DE` |
| device | `cuda` |
| chunk size | 250 |
| seed | 13 |
| train per label | 24,037 |
| validation per label | 758 |
| test per label | 205 |
| head train jobs | 8 |
| model types | `logistic_regression`, `random_forest`, `mlp` |
| persisted head models | `best` |

### PowerShell Launcher

The tracked launcher `scripts/run_local_50k.ps1` starts the local run outside
the chat session, redirects output, records a launcher PID, and can be rerun
safely.

### Test Coverage

The branch added recovery-specific tests for:

- local run state creation
- stale lock detection
- heartbeat writing
- resume command generation
- atomic output behavior
- sentence-head recovery
- skipping completed heads
- retrying incomplete heads
- merging final head metrics and predictions
- `persist-head-models=best`

The full test suite after the recovery implementation previously passed with
`21 passed`.

## 4. Feature Engineering Details

The extractor does not store raw attention matrices in the training feature
tables. Instead, it converts attention distributions into compact statistics.
This keeps the feature tables tractable while preserving interpretable signals.

### Attention Objects

Conceptually, for each example:

```text
cross_attentions[layer, head, target_token, source_token]
self_attentions[layer, head, target_token, target_token]
```

Cross-attention measures how generated target tokens attend to source tokens.
Decoder self-attention measures how each target token attends to previous or
available target-side positions under teacher forcing.

### Per-Distribution Statistics

For each attention distribution, the feature builder computes:

| statistic | interpretation |
|---|---|
| entropy | high values mean attention is diffuse |
| max | strongest single attention weight |
| variance | concentration/spread of the attention weights |
| top-k mass | total attention mass assigned to the top `k=3` positions |

For cross-attention, high entropy or low max can indicate weak source grounding.
For self-attention, high reliance on target context can indicate target-side
continuation behavior that is less directly grounded in the source.

### Ratio Features

The token and head features include:

| ratio | meaning |
|---|---|
| `self_to_cross_max_ratio` | compares strongest self-attention to strongest cross-attention |
| `self_to_cross_entropy_ratio` | compares diffuse self-attention to diffuse cross-attention |

These features are intended to expose whether a target token is behaving more
like a source-grounded translation token or like a continuation supported by the
target prefix.

### Token-Level Feature Rows

Each token feature row corresponds to one generated target token. The current
token classifier uses 45 feature columns.

Feature groups include:

| group | examples |
|---|---|
| cross-attention entropy | `cross_entropy_mean`, `cross_entropy_std`, `cross_entropy_min`, `cross_entropy_max`, `cross_entropy_last_layer_mean` |
| cross-attention max | `cross_max_mean`, `cross_max_std`, `cross_max_min`, `cross_max_max`, `cross_max_last_layer_mean` |
| cross-attention variance | `cross_variance_mean`, `cross_variance_std`, `cross_variance_min`, `cross_variance_max`, `cross_variance_last_layer_mean` |
| cross-attention top-k mass | `cross_topk_mass_mean`, `cross_topk_mass_std`, `cross_topk_mass_min`, `cross_topk_mass_max`, `cross_topk_mass_last_layer_mean` |
| self-attention equivalents | same families for `self_entropy`, `self_max`, `self_variance`, `self_topk_mass` |
| ratios | `self_to_cross_max_ratio`, `self_to_cross_entropy_ratio` |
| position/length | `token_relative_position`, `source_length_tokens`, `target_length_tokens` |

The token target is `token_label`.

### Sentence-Level Feature Rows

Each sentence feature row corresponds to one translated sentence. The current
sentence classifier uses 181 feature columns.

The sentence builder aggregates token-level features across target tokens. For
most token features it computes:

- sentence mean
- sentence max
- sentence min
- sentence standard deviation

The sentence target is `sentence_label`.

### Sentence-Head Feature Rows

Each sentence-head row corresponds to one sentence and one decoder
`(layer_id, head_id)`. mBART large has 12 decoder layers and 16 heads per
layer, so each sentence produces `12 * 16 = 192` sentence-head rows.

The current head classifier uses 36 feature columns:

| group | examples |
|---|---|
| cross entropy | mean/std/min/max aggregated across target tokens for that head |
| cross max | mean/std/min/max |
| cross variance | mean/std/min/max |
| cross top-k mass | mean/std/min/max |
| self entropy | mean/std/min/max |
| self max | mean/std/min/max |
| self variance | mean/std/min/max |
| self top-k mass | mean/std/min/max |
| ratios | `self_to_cross_max_ratio_mean`, `self_to_cross_entropy_ratio_mean` |
| length/head identity | `source_length_tokens`, `target_length_tokens`, `layer_id`, `head_id` |

The head classifier target is also `sentence_label`. The point is not to make a
separate production model for every head. The point is to rank which attention
heads carry the most predictive sentence-level hallucination signal.

### Why Teacher Forcing Matters

The attention features are extracted while the model reads the provided dataset
translation. That means the model is not generating a fresh hypothesis. The
features therefore describe how mBART attends when conditioned on the known
translation, including known hallucinated/errorful tokens. This is the right
setup for supervised hallucination localization and diagnosis, but it should
not be confused with measuring attention during free decoding.

## 5. Run Chronology

### 12-Example Smoke Run

Purpose: prove that the full feature extraction and classifier path worked with
real mBART on a tiny controlled subset.

The subset contained 2 positive and 2 negative sentence examples in each split.
All positive examples had token/span supervision. This run is correctness-only.
Its metrics are not scientific because each split contains only 4 examples.

### 1k Run

Purpose: establish a first practical baseline with real feature sizes and
train/validation/test splits.

The 1k run used 350 positive and 350 negative train examples, 75 positive and
75 negative validation examples, and 75 positive and 75 negative test examples.
It used logistic regression only.

This run showed that sentence-level metrics were already stronger than token
metrics. It also exposed the low positive token-label rate in validation/test,
which makes token localization difficult.

### 10k Reliable Scaling Run

Purpose: scale to 10,000 examples with chunked extraction and model comparison.

The 10k run used all available supervised positives in validation and test:
758 validation positives and 205 test positives. Because the original WMT test
split only provides 205 supervised positive examples, the run is balanced
overall rather than equally sized per split.

This was the first run with:

- chunked extraction
- 40 completed chunks
- logistic regression, random forest, and MLP comparisons
- complete head rankings for each model type
- final `run_summary.json`

### 50k Local Laptop Recovery Run

Purpose: run the current main scale step locally with recovery hardening.

The 50k run used 24,037 positive and 24,037 negative train examples, 758
positive and 758 negative validation examples, and 205 positive and 205
negative test examples.

This was the first run with:

- `lucid-run-local`
- durable `run_state.json`
- persistent logs
- heartbeat JSONL
- active/stale lock handling
- resumable chunked mBART extraction
- resumable sentence-head training
- `persist-head-models=best`
- 200 completed extraction chunks
- all three model families trained end to end

## 6. Dataset Composition

### High-Level Run Composition

| run | JSONL rows | split counts | labels 0/1 | supervised rows | duplicate rewrites |
|---|---:|---|---:|---:|---:|
| `smoke_12` | 12 | train:4, validation:4, test:4 | 6/6 | 6 | NA |
| `1k` | 1,000 | train:700, validation:150, test:150 | 500/500 | 500 | 19 |
| `10k` | 10,000 | train:8,074, validation:1,516, test:410 | 5,000/5,000 | 5,000 | 474 |
| `50k` | 50,000 | train:48,074, validation:1,516, test:410 | 25,000/25,000 | 25,000 | 2,958 |

`supervised rows` here means rows with token/span supervision. In these sampled
runs, that corresponds to the positive examples because negatives are selected
as sentence-level non-error rows without positive token spans.

### Full Sampler Availability

The sampler summaries for 1k, 10k, and 50k all came from the same full
trainable pool:

| sampler field | value |
|---|---:|
| input examples | 126,388 |
| available train negatives | 55,578 |
| available train supervised positives | 57,720 |
| available validation negatives | 1,926 |
| available validation supervised positives | 758 |
| available test negatives | 306 |
| available test supervised positives | 205 |
| positives skipped for missing token/span supervision | 9,895 |

The 10k and 50k runs are balanced overall, not equally sized per split, because
the test split cannot provide more than 205 supervised positive examples.

### Exact Split/Label/Supervision Composition

#### `smoke_12`

| split | sentence label | has token/span supervision | rows |
|---|---:|---|---:|
| test | 0 | False | 2 |
| test | 1 | True | 2 |
| train | 0 | False | 2 |
| train | 1 | True | 2 |
| validation | 0 | False | 2 |
| validation | 1 | True | 2 |

#### `1k`

| split | sentence label | has token/span supervision | rows |
|---|---:|---|---:|
| test | 0 | False | 75 |
| test | 1 | True | 75 |
| train | 0 | False | 350 |
| train | 1 | True | 350 |
| validation | 0 | False | 75 |
| validation | 1 | True | 75 |

#### `10k`

| split | sentence label | has token/span supervision | rows |
|---|---:|---|---:|
| test | 0 | False | 205 |
| test | 1 | True | 205 |
| train | 0 | False | 4,037 |
| train | 1 | True | 4,037 |
| validation | 0 | False | 758 |
| validation | 1 | True | 758 |

#### `50k`

| split | sentence label | has token/span supervision | rows |
|---|---:|---|---:|
| test | 0 | False | 205 |
| test | 1 | True | 205 |
| train | 0 | False | 24,037 |
| train | 1 | True | 24,037 |
| validation | 0 | False | 758 |
| validation | 1 | True | 758 |

### Token Label Prevalence

The sentence-level data is balanced by construction. The token-level data is
not balanced because a positive sentence generally contains a small number of
errorful tokens among many non-error tokens.

| run | split | token rows | positive token labels | positive rate |
|---|---|---:|---:|---:|
| 1k | train | 33,535 | 3,481 | 0.103802 |
| 1k | validation | 5,831 | 352 | 0.060367 |
| 1k | test | 5,733 | 384 | 0.066981 |
| 10k | train | 378,567 | 42,642 | 0.112641 |
| 10k | validation | 56,204 | 3,528 | 0.062771 |
| 10k | test | 15,690 | 1,069 | 0.068133 |
| 50k | train | 2,222,611 | 260,119 | 0.117033 |
| 50k | validation | 56,204 | 3,528 | 0.062771 |
| 50k | test | 15,690 | 1,069 | 0.068133 |

This is a central reason token precision is low. On the 50k test set, only
6.8133% of token rows are positive. A model can have reasonable recall while
still generating many false positives.

### Sentence And Token Lengths

The target token counts explain why token rows grow much faster than sentence
rows.

| run | split | sentences | target tokens mean | target tokens min | target tokens max | source length mean |
|---|---|---:|---:|---:|---:|---:|
| smoke_12 | train | 4 | 2.000 | 2 | 2 | 4.500 |
| smoke_12 | validation | 4 | 6.000 | 4 | 8 | 6.500 |
| smoke_12 | test | 4 | 6.000 | 5 | 7 | 8.250 |
| 1k | train | 700 | 47.907 | 3 | 232 | 41.199 |
| 1k | validation | 150 | 38.873 | 7 | 95 | 35.173 |
| 1k | test | 150 | 38.220 | 5 | 105 | 36.193 |
| 10k | train | 8,074 | 46.887 | 2 | 254 | 40.231 |
| 10k | validation | 1,516 | 37.074 | 4 | 104 | 33.643 |
| 10k | test | 410 | 38.268 | 5 | 117 | 35.637 |
| 50k | train | 48,074 | 46.233 | 2 | 254 | 39.643 |
| 50k | validation | 1,516 | 37.074 | 4 | 104 | 33.643 |
| 50k | test | 410 | 38.268 | 5 | 117 | 35.637 |

## 7. Extraction Results

### Extraction Summary

| run | processed | skipped | token rows | sentence rows | sentence-head rows | chunks | parquet token/sentence/head rows |
|---|---:|---:|---:|---:|---:|---:|---|
| `smoke_12` | 12 | 0 | 56 | 12 | 2,304 | 0 | 56 / 12 / 2,304 |
| `1k` | 1,000 | 0 | 45,099 | 1,000 | 192,000 | 0 | 45,099 / 1,000 / 192,000 |
| `10k` | 10,000 | 0 | 450,461 | 10,000 | 1,920,000 | 40 | 450,461 / 10,000 / 1,920,000 |
| `50k` | 50,000 | 0 | 2,294,505 | 50,000 | 9,600,000 | 200 | 2,294,505 / 50,000 / 9,600,000 |

The sentence-head row counts match the expected formula:

```text
sentence_head_rows = sentence_rows * 12 decoder layers * 16 heads
```

Examples:

```text
50,000 * 192 = 9,600,000
10,000 * 192 = 1,920,000
1,000 * 192 = 192,000
12 * 192 = 2,304
```

### 50k Feature File Sizes

| file | size bytes | last write |
|---|---:|---|
| `data/processed/en_de_50k/token_features.parquet` | 783,648,206 | 2026-05-05 3:20 PM |
| `data/processed/en_de_50k/sentence_features.parquet` | 60,150,933 | 2026-05-05 3:20 PM |
| `data/processed/en_de_50k/sentence_head_features.parquet` | 2,505,766,866 | 2026-05-05 3:21 PM |
| `artifacts/en_de_50k/run_summary.json` | 131,305 | 2026-05-05 5:25 PM |
| `artifacts/en_de_50k/logs/heartbeat.jsonl` | 362,041 | 2026-05-05 5:25 PM |

### Extraction Interpretation

The extraction path scaled cleanly:

- no skipped examples in any run
- 40/40 completed chunks for 10k
- 200/200 completed chunks for 50k
- final parquet metadata row counts match the JSON extraction reports
- validation and test row counts are identical between 10k and 50k because
  those splits used the same maximum available supervised-positive counts
- most of the 10k to 50k increase is in training rows

The 50k extraction took by far the most time. It was the only stage spanning
more than one calendar day.

## 8. Classifier Results

### Notes On Metrics

The training metrics JSON stores precision, recall, F1, ROC-AUC, and selected
threshold for validation and test. The top-level artifacts persist
`test_predictions.parquet`, so test confusion matrices were recomputed from
prediction files. Validation prediction files are not persisted in the same
top-level artifact layout, so validation confusion matrices are reported as
not available in this document.

For sentence-head classifiers, the top-level `test_predictions.parquet`
contains one row per test sentence per head. The best-head confusion matrix is
computed by filtering that file to the selected best `(layer_id, head_id)`.

### Token And Sentence Classifier Metrics

| run | model | classifier | split | precision | recall | F1 | ROC-AUC | threshold | prediction rows | confusion matrix |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| smoke_12 | logistic_regression | token | validation | 0.500000 | 1.000000 | 0.666667 | 0.500000 | 0.050000 | NA | NA |
| smoke_12 | logistic_regression | token | test | 0.333333 | 1.000000 | 0.500000 | 0.500000 | 0.050000 | 12 | tn=0, fp=8, fn=0, tp=4 |
| smoke_12 | logistic_regression | sentence | validation | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.050000 | NA | NA |
| smoke_12 | logistic_regression | sentence | test | 0.000000 | 0.000000 | 0.000000 | 0.500000 | 0.050000 | 4 | tn=2, fp=0, fn=2, tp=0 |
| 1k | logistic_regression | token | validation | 0.067910 | 0.403409 | 0.116251 | 0.531143 | 0.500000 | NA | NA |
| 1k | logistic_regression | token | test | 0.082726 | 0.414062 | 0.137901 | 0.560158 | 0.500000 | 5,733 | tn=3586, fp=1763, fn=225, tp=159 |
| 1k | logistic_regression | sentence | validation | 0.510638 | 0.960000 | 0.666667 | 0.532267 | 0.100000 | NA | NA |
| 1k | logistic_regression | sentence | test | 0.523077 | 0.906667 | 0.663415 | 0.585956 | 0.100000 | 150 | tn=13, fp=62, fn=7, tp=68 |
| 10k | logistic_regression | token | validation | 0.076074 | 0.484410 | 0.131497 | 0.564804 | 0.500000 | NA | NA |
| 10k | logistic_regression | token | test | 0.080000 | 0.434051 | 0.135100 | 0.555489 | 0.500000 | 15,690 | tn=9285, fp=5336, fn=605, tp=464 |
| 10k | logistic_regression | sentence | validation | 0.569686 | 0.862797 | 0.686254 | 0.674243 | 0.350000 | NA | NA |
| 10k | logistic_regression | sentence | test | 0.583630 | 0.800000 | 0.674897 | 0.658108 | 0.350000 | 410 | tn=88, fp=117, fn=41, tp=164 |
| 10k | random_forest | token | validation | 0.070487 | 0.675737 | 0.127657 | 0.551976 | 0.100000 | NA | NA |
| 10k | random_forest | token | test | 0.075192 | 0.668849 | 0.135186 | 0.553131 | 0.100000 | 15,690 | tn=5827, fp=8794, fn=354, tp=715 |
| 10k | random_forest | sentence | validation | 0.542254 | 0.914248 | 0.680747 | 0.632486 | 0.350000 | NA | NA |
| 10k | random_forest | sentence | test | 0.547988 | 0.863415 | 0.670455 | 0.647948 | 0.350000 | 410 | tn=59, fp=146, fn=28, tp=177 |
| 10k | mlp | token | validation | 0.074700 | 0.547336 | 0.131459 | 0.569384 | 0.100000 | NA | NA |
| 10k | mlp | token | test | 0.078207 | 0.543499 | 0.136738 | 0.558998 | 0.100000 | 15,690 | tn=7773, fp=6848, fn=488, tp=581 |
| 10k | mlp | sentence | validation | 0.546012 | 0.704485 | 0.615207 | 0.589495 | 0.050000 | NA | NA |
| 10k | mlp | sentence | test | 0.503937 | 0.624390 | 0.557734 | 0.544343 | 0.050000 | 410 | tn=79, fp=126, fn=77, tp=128 |
| 50k | logistic_regression | token | validation | 0.078394 | 0.426587 | 0.132447 | 0.562841 | 0.500000 | NA | NA |
| 50k | logistic_regression | token | test | 0.081616 | 0.370440 | 0.133761 | 0.555719 | 0.500000 | 15,690 | tn=10165, fp=4456, fn=673, tp=396 |
| 50k | logistic_regression | sentence | validation | 0.572552 | 0.864116 | 0.688749 | 0.678376 | 0.350000 | NA | NA |
| 50k | logistic_regression | sentence | test | 0.618705 | 0.839024 | 0.712215 | 0.695324 | 0.350000 | 410 | tn=99, fp=106, fn=33, tp=172 |
| 50k | random_forest | token | validation | 0.079989 | 0.398243 | 0.133220 | 0.574712 | 0.150000 | NA | NA |
| 50k | random_forest | token | test | 0.083613 | 0.372311 | 0.136559 | 0.558880 | 0.150000 | 15,690 | tn=10259, fp=4362, fn=671, tp=398 |
| 50k | random_forest | sentence | validation | 0.513680 | 0.990765 | 0.676577 | 0.657846 | 0.250000 | NA | NA |
| 50k | random_forest | sentence | test | 0.507653 | 0.970732 | 0.666667 | 0.668769 | 0.250000 | 410 | tn=12, fp=193, fn=6, tp=199 |
| 50k | mlp | token | validation | 0.077979 | 0.637472 | 0.138960 | 0.594038 | 0.100000 | NA | NA |
| 50k | mlp | token | test | 0.084082 | 0.656688 | 0.149076 | 0.596523 | 0.100000 | 15,690 | tn=6974, fp=7647, fn=367, tp=702 |
| 50k | mlp | sentence | validation | 0.529839 | 0.866755 | 0.657658 | 0.608024 | 0.050000 | NA | NA |
| 50k | mlp | sentence | test | 0.536585 | 0.858537 | 0.660413 | 0.608733 | 0.050000 | 410 | tn=53, fp=152, fn=29, tp=176 |

### Best Sentence-Head Metrics

| run | model | trained heads | best layer | best head | rank score | val precision | val recall | val F1 | val ROC-AUC | test precision | test recall | test F1 | test ROC-AUC | threshold | best pred rows | best confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| smoke_12 | logistic_regression | 192 | 6 | 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.100000 | 4 | tn=2, fp=0, fn=0, tp=2 |
| 1k | logistic_regression | 192 | 7 | 3 | 0.681422 | 0.528986 | 0.973333 | 0.685446 | 0.681422 | 0.537313 | 0.960000 | 0.688995 | 0.709867 | 0.200000 | 150 | tn=13, fp=62, fn=3, tp=72 |
| 10k | logistic_regression | 192 | 6 | 13 | 0.713472 | 0.568163 | 0.918206 | 0.701967 | 0.713472 | 0.590323 | 0.892683 | 0.710680 | 0.726901 | 0.350000 | 410 | tn=78, fp=127, fn=22, tp=183 |
| 10k | random_forest | 192 | 9 | 13 | 0.686186 | 0.557845 | 0.928760 | 0.697030 | 0.686186 | 0.548287 | 0.858537 | 0.669202 | 0.652278 | 0.350000 | 410 | tn=60, fp=145, fn=29, tp=176 |
| 10k | mlp | 192 | 6 | 13 | 0.658779 | 0.556514 | 0.850923 | 0.672926 | 0.658779 | 0.545741 | 0.843902 | 0.662835 | 0.626889 | 0.250000 | 410 | tn=61, fp=144, fn=32, tp=173 |
| 50k | logistic_regression | 192 | 6 | 13 | 0.717134 | 0.566038 | 0.910290 | 0.698027 | 0.717134 | 0.588997 | 0.887805 | 0.708171 | 0.728447 | 0.350000 | 410 | tn=78, fp=127, fn=23, tp=182 |
| 50k | random_forest | 192 | 7 | 3 | 0.692411 | 0.578901 | 0.875989 | 0.697113 | 0.692411 | 0.592466 | 0.843902 | 0.696177 | 0.726651 | 0.400000 | 410 | tn=86, fp=119, fn=32, tp=173 |
| 50k | mlp | 192 | 9 | 13 | 0.645595 | 0.525127 | 0.951187 | 0.676678 | 0.645595 | 0.517241 | 0.951220 | 0.670103 | 0.650732 | 0.100000 | 410 | tn=23, fp=182, fn=10, tp=195 |

### Top 10 Heads And Distributions

#### `smoke_12` logistic regression heads

The smoke head results are correctness-only. With 4 examples per split, perfect
metrics are not meaningful as scientific evidence.

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 3 | 0.100000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 5 | 9 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 0.750000 |
| 3 | 5 | 14 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 1.000000 |
| 4 | 8 | 3 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.666667 | 1.000000 | 0.800000 | 0.750000 |
| 5 | 0 | 5 | 0.600000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 0.750000 |
| 6 | 3 | 4 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 1.000000 |
| 7 | 3 | 5 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 0.500000 |
| 8 | 4 | 3 | 0.350000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 0.750000 |
| 9 | 4 | 7 | 0.050000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 0.666667 | 0.500000 |
| 10 | 5 | 4 | 0.650000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.500000 | 1.000000 | 0.666667 | 0.500000 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 | 0.498698 |
| test ROC-AUC | 0.000000 | 0.500000 | 0.500000 | 0.625000 | 1.000000 | 0.530599 |
| validation F1 | 0.000000 | 0.000000 | 0.666667 | 0.666667 | 1.000000 | 0.460069 |
| test F1 | 0.000000 | 0.000000 | 0.666667 | 0.666667 | 1.000000 | 0.435417 |

#### `1k` logistic regression heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 3 | 0.200000 | 0.528986 | 0.973333 | 0.685446 | 0.681422 | 0.537313 | 0.960000 | 0.688995 | 0.709867 |
| 2 | 4 | 4 | 0.250000 | 0.549618 | 0.960000 | 0.699029 | 0.674844 | 0.533835 | 0.946667 | 0.682692 | 0.651200 |
| 3 | 4 | 9 | 0.400000 | 0.587629 | 0.760000 | 0.662791 | 0.663289 | 0.588235 | 0.666667 | 0.625000 | 0.664533 |
| 4 | 9 | 3 | 0.350000 | 0.555556 | 0.866667 | 0.677083 | 0.640889 | 0.513043 | 0.786667 | 0.621053 | 0.636089 |
| 5 | 8 | 3 | 0.300000 | 0.538462 | 0.933333 | 0.682927 | 0.640711 | 0.536000 | 0.893333 | 0.670000 | 0.613867 |
| 6 | 0 | 8 | 0.300000 | 0.560345 | 0.866667 | 0.680628 | 0.635022 | 0.561404 | 0.853333 | 0.677249 | 0.699911 |
| 7 | 8 | 13 | 0.300000 | 0.558333 | 0.893333 | 0.687179 | 0.632000 | 0.541667 | 0.866667 | 0.666667 | 0.620089 |
| 8 | 8 | 12 | 0.300000 | 0.563025 | 0.893333 | 0.690722 | 0.631822 | 0.550459 | 0.800000 | 0.652174 | 0.631467 |
| 9 | 10 | 11 | 0.350000 | 0.584906 | 0.826667 | 0.685083 | 0.630400 | 0.608247 | 0.786667 | 0.686047 | 0.620800 |
| 10 | 0 | 14 | 0.300000 | 0.546875 | 0.933333 | 0.689655 | 0.630222 | 0.577982 | 0.840000 | 0.684783 | 0.647822 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.476089 | 0.538133 | 0.566044 | 0.592533 | 0.681422 | 0.565244 |
| test ROC-AUC | 0.558400 | 0.604444 | 0.633244 | 0.656533 | 0.732089 | 0.632639 |
| validation F1 | 0.648649 | 0.666667 | 0.669767 | 0.676471 | 0.707071 | 0.672460 |
| test F1 | 0.604396 | 0.663462 | 0.666667 | 0.675799 | 0.711340 | 0.668341 |

#### `10k` logistic regression heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 13 | 0.350000 | 0.568163 | 0.918206 | 0.701967 | 0.713472 | 0.590323 | 0.892683 | 0.710680 | 0.726901 |
| 2 | 9 | 13 | 0.350000 | 0.570724 | 0.915567 | 0.703141 | 0.712407 | 0.567657 | 0.839024 | 0.677165 | 0.707983 |
| 3 | 7 | 3 | 0.350000 | 0.564503 | 0.906332 | 0.695696 | 0.703994 | 0.593548 | 0.897561 | 0.714563 | 0.732183 |
| 4 | 6 | 9 | 0.350000 | 0.553686 | 0.911609 | 0.688933 | 0.691398 | 0.575385 | 0.912195 | 0.705660 | 0.702439 |
| 5 | 4 | 9 | 0.350000 | 0.568644 | 0.885224 | 0.692466 | 0.688575 | 0.578767 | 0.824390 | 0.680080 | 0.697061 |
| 6 | 8 | 7 | 0.350000 | 0.567114 | 0.891821 | 0.693333 | 0.687691 | 0.592949 | 0.902439 | 0.715667 | 0.708911 |
| 7 | 10 | 13 | 0.300000 | 0.544747 | 0.923483 | 0.685267 | 0.684214 | 0.559271 | 0.897561 | 0.689139 | 0.693111 |
| 8 | 9 | 12 | 0.350000 | 0.554750 | 0.908971 | 0.689000 | 0.683733 | 0.563467 | 0.887805 | 0.689394 | 0.715622 |
| 9 | 6 | 14 | 0.300000 | 0.548289 | 0.951187 | 0.695610 | 0.682948 | 0.570149 | 0.931707 | 0.707407 | 0.731350 |
| 10 | 7 | 1 | 0.300000 | 0.543939 | 0.947230 | 0.691049 | 0.682873 | 0.555556 | 0.926829 | 0.694698 | 0.696443 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.606146 | 0.635883 | 0.651136 | 0.667603 | 0.713472 | 0.651069 |
| test ROC-AUC | 0.620845 | 0.663986 | 0.680024 | 0.693230 | 0.738203 | 0.679326 |
| validation F1 | 0.670631 | 0.681329 | 0.685301 | 0.689829 | 0.703141 | 0.685341 |
| test F1 | 0.660959 | 0.676636 | 0.682310 | 0.688525 | 0.715667 | 0.683146 |

#### `10k` random forest heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 13 | 0.350000 | 0.557845 | 0.928760 | 0.697030 | 0.686186 | 0.548287 | 0.858537 | 0.669202 | 0.652278 |
| 2 | 7 | 3 | 0.300000 | 0.535874 | 0.945910 | 0.684160 | 0.685572 | 0.558140 | 0.936585 | 0.699454 | 0.727912 |
| 3 | 6 | 13 | 0.350000 | 0.557042 | 0.934037 | 0.697881 | 0.676821 | 0.582803 | 0.892683 | 0.705202 | 0.711220 |
| 4 | 6 | 5 | 0.350000 | 0.550079 | 0.912929 | 0.686508 | 0.674981 | 0.551205 | 0.892683 | 0.681564 | 0.676788 |
| 5 | 7 | 1 | 0.300000 | 0.532138 | 0.939314 | 0.679389 | 0.674242 | 0.524079 | 0.902439 | 0.663082 | 0.673206 |
| 6 | 6 | 7 | 0.400000 | 0.574318 | 0.861478 | 0.689182 | 0.673841 | 0.570934 | 0.804878 | 0.668016 | 0.681273 |
| 7 | 6 | 14 | 0.250000 | 0.530347 | 0.968338 | 0.685341 | 0.668227 | 0.540390 | 0.946341 | 0.687943 | 0.689256 |
| 8 | 8 | 7 | 0.350000 | 0.556726 | 0.906332 | 0.689759 | 0.666194 | 0.584375 | 0.912195 | 0.712381 | 0.710934 |
| 9 | 10 | 13 | 0.300000 | 0.535527 | 0.944591 | 0.683532 | 0.662441 | 0.548851 | 0.931707 | 0.690778 | 0.672469 |
| 10 | 5 | 0 | 0.250000 | 0.519068 | 0.969657 | 0.676173 | 0.661478 | 0.522788 | 0.951220 | 0.674740 | 0.674277 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.579679 | 0.612256 | 0.624731 | 0.643112 | 0.686186 | 0.627565 |
| test ROC-AUC | 0.599607 | 0.648459 | 0.660928 | 0.673908 | 0.727912 | 0.661092 |
| validation F1 | 0.668144 | 0.672745 | 0.676295 | 0.679400 | 0.697881 | 0.676747 |
| test F1 | 0.656085 | 0.668908 | 0.674576 | 0.683274 | 0.716698 | 0.676003 |

#### `10k` MLP heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 13 | 0.250000 | 0.556514 | 0.850923 | 0.672926 | 0.658779 | 0.545741 | 0.843902 | 0.662835 | 0.626889 |
| 2 | 6 | 9 | 0.050000 | 0.550120 | 0.905013 | 0.684289 | 0.647255 | 0.541796 | 0.853659 | 0.662879 | 0.628840 |
| 3 | 5 | 14 | 0.050000 | 0.530628 | 0.914248 | 0.671512 | 0.638081 | 0.512605 | 0.892683 | 0.651246 | 0.593813 |
| 4 | 10 | 13 | 0.050000 | 0.541738 | 0.839050 | 0.658385 | 0.628493 | 0.554140 | 0.848780 | 0.670520 | 0.609613 |
| 5 | 7 | 3 | 0.050000 | 0.528638 | 0.901055 | 0.666341 | 0.620746 | 0.539589 | 0.897561 | 0.673993 | 0.648400 |
| 6 | 9 | 8 | 0.050000 | 0.527755 | 0.865435 | 0.655672 | 0.620059 | 0.536585 | 0.858537 | 0.660413 | 0.613111 |
| 7 | 8 | 12 | 0.050000 | 0.519937 | 0.877309 | 0.652921 | 0.619274 | 0.521614 | 0.882927 | 0.655797 | 0.602570 |
| 8 | 9 | 4 | 0.050000 | 0.525331 | 0.889182 | 0.660461 | 0.617312 | 0.520349 | 0.873171 | 0.652095 | 0.582915 |
| 9 | 7 | 4 | 0.050000 | 0.536349 | 0.875989 | 0.665331 | 0.614371 | 0.536278 | 0.829268 | 0.651341 | 0.616109 |
| 10 | 11 | 1 | 0.050000 | 0.534387 | 0.891821 | 0.668314 | 0.610311 | 0.524096 | 0.848780 | 0.648045 | 0.599833 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.511631 | 0.555251 | 0.568343 | 0.585574 | 0.658779 | 0.570751 |
| test ROC-AUC | 0.494777 | 0.557763 | 0.580535 | 0.598572 | 0.656895 | 0.578479 |
| validation F1 | 0.616803 | 0.642787 | 0.651816 | 0.660194 | 0.684289 | 0.650877 |
| test F1 | 0.593625 | 0.646957 | 0.655303 | 0.663194 | 0.681481 | 0.653404 |

#### `50k` logistic regression heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 13 | 0.350000 | 0.566038 | 0.910290 | 0.698027 | 0.717134 | 0.588997 | 0.887805 | 0.708171 | 0.728447 |
| 2 | 9 | 13 | 0.350000 | 0.572848 | 0.912929 | 0.703967 | 0.712814 | 0.564356 | 0.834146 | 0.673228 | 0.713670 |
| 3 | 7 | 3 | 0.350000 | 0.573322 | 0.912929 | 0.704326 | 0.706086 | 0.595469 | 0.897561 | 0.715953 | 0.735348 |
| 4 | 6 | 9 | 0.400000 | 0.577739 | 0.862797 | 0.692063 | 0.695690 | 0.591549 | 0.819512 | 0.687117 | 0.707507 |
| 5 | 9 | 12 | 0.350000 | 0.562810 | 0.898417 | 0.692073 | 0.689851 | 0.566343 | 0.853659 | 0.680934 | 0.712409 |
| 6 | 7 | 1 | 0.300000 | 0.538751 | 0.944591 | 0.686152 | 0.688453 | 0.552632 | 0.921951 | 0.691042 | 0.702963 |
| 7 | 4 | 9 | 0.350000 | 0.575862 | 0.881266 | 0.696559 | 0.687559 | 0.589655 | 0.834146 | 0.690909 | 0.697371 |
| 8 | 10 | 13 | 0.300000 | 0.547656 | 0.924802 | 0.687929 | 0.686065 | 0.557864 | 0.917073 | 0.693727 | 0.695848 |
| 9 | 9 | 6 | 0.300000 | 0.549350 | 0.947230 | 0.695400 | 0.685223 | 0.550432 | 0.931707 | 0.692029 | 0.709935 |
| 10 | 8 | 7 | 0.350000 | 0.567747 | 0.901055 | 0.696583 | 0.684355 | 0.596154 | 0.907317 | 0.719536 | 0.715526 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.610157 | 0.637483 | 0.652454 | 0.668095 | 0.717134 | 0.652498 |
| test ROC-AUC | 0.624557 | 0.668031 | 0.684355 | 0.696990 | 0.741297 | 0.682475 |
| validation F1 | 0.670648 | 0.681420 | 0.685687 | 0.690052 | 0.704326 | 0.685830 |
| test F1 | 0.653465 | 0.677477 | 0.683019 | 0.691042 | 0.719536 | 0.683657 |

#### `50k` random forest heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 3 | 0.400000 | 0.578901 | 0.875989 | 0.697113 | 0.692411 | 0.592466 | 0.843902 | 0.696177 | 0.726651 |
| 2 | 6 | 13 | 0.400000 | 0.566007 | 0.905013 | 0.696447 | 0.692292 | 0.587629 | 0.834146 | 0.689516 | 0.701951 |
| 3 | 9 | 13 | 0.350000 | 0.552693 | 0.934037 | 0.694458 | 0.688544 | 0.551320 | 0.917073 | 0.688645 | 0.701927 |
| 4 | 8 | 7 | 0.450000 | 0.599418 | 0.815303 | 0.690889 | 0.679547 | 0.620155 | 0.780488 | 0.691145 | 0.700262 |
| 5 | 8 | 1 | 0.400000 | 0.574154 | 0.873351 | 0.692831 | 0.678811 | 0.585616 | 0.834146 | 0.688129 | 0.683093 |
| 6 | 10 | 13 | 0.400000 | 0.573555 | 0.864116 | 0.689474 | 0.678262 | 0.585034 | 0.839024 | 0.689379 | 0.693540 |
| 7 | 6 | 9 | 0.400000 | 0.560771 | 0.882586 | 0.685802 | 0.675508 | 0.561728 | 0.887805 | 0.688091 | 0.704128 |
| 8 | 6 | 7 | 0.400000 | 0.562446 | 0.861478 | 0.680563 | 0.675348 | 0.568627 | 0.848780 | 0.681018 | 0.691600 |
| 9 | 7 | 1 | 0.300000 | 0.524555 | 0.972296 | 0.681461 | 0.675136 | 0.510870 | 0.917073 | 0.656195 | 0.665057 |
| 10 | 5 | 12 | 0.350000 | 0.550319 | 0.908971 | 0.685572 | 0.674879 | 0.541176 | 0.897561 | 0.675229 | 0.669197 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.576905 | 0.616775 | 0.631143 | 0.648547 | 0.692411 | 0.632715 |
| test ROC-AUC | 0.619203 | 0.650339 | 0.666020 | 0.678477 | 0.726651 | 0.665605 |
| validation F1 | 0.666961 | 0.672245 | 0.676306 | 0.680563 | 0.697113 | 0.676980 |
| test F1 | 0.633858 | 0.668977 | 0.675090 | 0.682594 | 0.704120 | 0.676123 |

#### `50k` MLP heads

| rank | layer | head | threshold | val P | val R | val F1 | val AUC | test P | test R | test F1 | test AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 13 | 0.100000 | 0.525127 | 0.951187 | 0.676678 | 0.645595 | 0.517241 | 0.951220 | 0.670103 | 0.650732 |
| 2 | 7 | 1 | 0.100000 | 0.524088 | 0.947230 | 0.674812 | 0.639578 | 0.518717 | 0.946341 | 0.670121 | 0.651898 |
| 3 | 6 | 5 | 0.100000 | 0.519537 | 0.947230 | 0.671028 | 0.639260 | 0.522788 | 0.951220 | 0.674740 | 0.623058 |
| 4 | 6 | 2 | 0.050000 | 0.511236 | 0.960422 | 0.667278 | 0.631685 | 0.523438 | 0.980488 | 0.682513 | 0.588198 |
| 5 | 6 | 13 | 0.300000 | 0.539872 | 0.893140 | 0.672962 | 0.631246 | 0.552147 | 0.878049 | 0.677966 | 0.652659 |
| 6 | 6 | 15 | 0.050000 | 0.507483 | 0.984169 | 0.669659 | 0.630002 | 0.508685 | 1.000000 | 0.674342 | 0.635955 |
| 7 | 6 | 7 | 0.050000 | 0.512838 | 0.974934 | 0.672124 | 0.629823 | 0.510256 | 0.970732 | 0.668908 | 0.621797 |
| 8 | 8 | 15 | 0.100000 | 0.518438 | 0.945910 | 0.669780 | 0.628827 | 0.512064 | 0.931707 | 0.660900 | 0.551315 |
| 9 | 7 | 3 | 0.100000 | 0.513514 | 0.952507 | 0.667283 | 0.626798 | 0.517060 | 0.960976 | 0.672355 | 0.656681 |
| 10 | 8 | 13 | 0.100000 | 0.523775 | 0.944591 | 0.673882 | 0.625381 | 0.524862 | 0.926829 | 0.670194 | 0.624105 |

| metric | min | p25 | median | p75 | max | mean |
|---|---:|---:|---:|---:|---:|---:|
| validation ROC-AUC | 0.530816 | 0.573244 | 0.586583 | 0.603006 | 0.645595 | 0.587075 |
| test ROC-AUC | 0.524402 | 0.583319 | 0.604497 | 0.625747 | 0.689637 | 0.604403 |
| validation F1 | 0.652476 | 0.665458 | 0.668521 | 0.672230 | 0.682309 | 0.668667 |
| test F1 | 0.627949 | 0.664441 | 0.669983 | 0.673759 | 0.701627 | 0.668989 |

## 9. Metric Interpretation

### Token Localization Is Still Weak

The token classifiers are the weakest part of the pipeline.

At 50k:

| model | test precision | test recall | test F1 | test ROC-AUC |
|---|---:|---:|---:|---:|
| logistic regression | 0.081616 | 0.370440 | 0.133761 | 0.555719 |
| random forest | 0.083613 | 0.372311 | 0.136559 | 0.558880 |
| MLP | 0.084082 | 0.656688 | 0.149076 | 0.596523 |

The MLP has the best token F1 and ROC-AUC at 50k, but the absolute precision is
still only `0.084082`. That means most positive token predictions are false
positives. This is not surprising given the test positive token rate of only
`0.068133`. The classifier is operating in a heavily imbalanced token-level
problem even though the sentence-level dataset is balanced.

The 50k MLP token model is still useful as a signal that more training data and
nonlinear modeling can improve token ranking. It is not yet a reliable
localizer.

### Sentence-Level Prediction Is Stronger

The sentence classifiers have a much easier label structure: one balanced label
per sentence. The 50k logistic regression sentence classifier is the strongest
whole-sentence model:

| metric | 10k logistic sentence | 50k logistic sentence | delta |
|---|---:|---:|---:|
| test F1 | 0.674897 | 0.712215 | +0.037318 |
| test ROC-AUC | 0.658108 | 0.695324 | +0.037216 |

The 50k logistic sentence confusion matrix is:

```text
tn=99, fp=106, fn=33, tp=172
```

This means it catches most positives (`recall=0.839024`) while keeping false
positives lower than random forest at the chosen threshold.

The 50k random forest sentence model has very high recall:

```text
recall=0.970732, precision=0.507653, tn=12, fp=193, fn=6, tp=199
```

That model is useful if missing hallucinations is far worse than overflagging,
but its threshold currently produces too many false positives for a balanced
decision setting.

### Sentence-Head Models Are Predictive And Interpretable

The sentence-head experiments show that individual mBART decoder attention
heads contain sentence-level hallucination/error signal.

The most stable logistic-regression head is:

```text
Layer 6 Head 13
```

It was best at both 10k and 50k:

| run | model | best head | validation ROC-AUC | test ROC-AUC | test F1 |
|---|---|---|---:|---:|---:|
| 10k | logistic regression | L6 H13 | 0.713472 | 0.726901 | 0.710680 |
| 50k | logistic regression | L6 H13 | 0.717134 | 0.728447 | 0.708171 |

The test ROC-AUC is extremely stable across scale:

```text
0.726901 -> 0.728447, delta +0.001547
```

The small F1 decrease:

```text
0.710680 -> 0.708171, delta -0.002508
```

is not concerning because the validation ranking score and test ROC-AUC both
remain strong, and the test set is identical between the 10k and 50k runs.

Other recurring heads:

| head | where it appears |
|---|---|
| L7 H3 | best in 1k logistic, rank 3 in 10k logistic, rank 3 in 50k logistic, best in 50k random forest |
| L9 H13 | rank 2 in 10k logistic, rank 1 in 10k random forest, rank 2 in 50k logistic, rank 3 in 50k random forest, best in 50k MLP |
| L6 H9 | top 10 in 10k and 50k logistic/random forest |
| L8 H7 | top 10 in 10k and 50k logistic/random forest |

The repeated presence of heads around layers 6 to 10 suggests the useful
sentence-level attention signal is concentrated in middle-to-late decoder
layers, not uniformly distributed across all 192 heads.

### 10k To 50k Comparison

| model | classifier | 10k test F1 | 50k test F1 | F1 delta | 10k test ROC-AUC | 50k test ROC-AUC | AUC delta |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic regression | token | 0.135100 | 0.133761 | -0.001339 | 0.555489 | 0.555719 | +0.000230 |
| random forest | token | 0.135186 | 0.136559 | +0.001372 | 0.553131 | 0.558880 | +0.005749 |
| MLP | token | 0.136738 | 0.149076 | +0.012338 | 0.558998 | 0.596523 | +0.037525 |
| logistic regression | sentence | 0.674897 | 0.712215 | +0.037318 | 0.658108 | 0.695324 | +0.037216 |
| random forest | sentence | 0.670455 | 0.666667 | -0.003788 | 0.647948 | 0.668769 | +0.020821 |
| MLP | sentence | 0.557734 | 0.660413 | +0.102679 | 0.544343 | 0.608733 | +0.064390 |
| logistic regression | sentence-head | 0.710680 | 0.708171 | -0.002508 | 0.726901 | 0.728447 | +0.001547 |
| random forest | sentence-head | 0.669202 | 0.696177 | +0.026976 | 0.652278 | 0.726651 | +0.074372 |
| MLP | sentence-head | 0.662835 | 0.670103 | +0.007268 | 0.626889 | 0.650732 | +0.023843 |

Main conclusions:

- 50k clearly improved the logistic sentence classifier.
- 50k clearly improved the MLP token classifier, though token absolute quality
  remains weak.
- 50k strongly improved random forest sentence-head ROC-AUC.
- Logistic sentence-head performance was already strong at 10k and stayed
  stable at 50k.
- Adding more training examples alone did not solve token localization.

### Smoke Run Interpretation

The smoke run must be treated only as pipeline-correctness evidence.

Reasons:

- 4 train examples
- 4 validation examples
- 4 test examples
- 2 positive examples per split
- tiny token row counts
- many head classifiers can look perfect by chance

The smoke run proved that data loading, feature extraction, feature validation,
training, metrics, prediction writing, and head ranking all connected. It does
not provide evidence about model quality.

## 10. 50k Runtime And Recovery Analysis

### Environment

The 50k run wrote `artifacts/en_de_50k/environment.json` with:

| field | value |
|---|---|
| checked at | 2026-05-04T16:59:23+00:00 |
| python | `.venv\Scripts\python.exe` |
| platform | Windows-11-10.0.26200-SP0 |
| CUDA available | true |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| torch | 2.11.0+cu128 |
| transformers | 5.7.0 |
| sklearn | 1.8.0 |
| pyarrow | 24.0.0 |

CUDA was required for this run. The orchestrator would have stopped if CUDA was
unavailable or if the GPU name did not match the expected RTX 4070 Laptop GPU.

### Runtime Timeline

The 50k run started at `2026-05-04T16:59:23+00:00` and completed at
`2026-05-05T21:25:34+00:00`, for a total wall-clock duration of about
`28:26:11`.

| stage | started UTC | completed UTC | duration |
|---|---|---|---:|
| env-check | 2026-05-04T16:59:23+00:00 | 2026-05-04T16:59:33+00:00 | 00:00:10 |
| sample | 2026-05-04T16:59:33+00:00 | 2026-05-04T17:00:03+00:00 | 00:00:30 |
| extract | 2026-05-04T17:00:03+00:00 | 2026-05-05T19:21:56+00:00 | 26:21:53 |
| train-token:logistic_regression | 2026-05-05T19:21:56+00:00 | 2026-05-05T19:22:56+00:00 | 00:01:00 |
| train-sentence:logistic_regression | 2026-05-05T19:22:56+00:00 | 2026-05-05T19:23:26+00:00 | 00:00:30 |
| train-heads:logistic_regression | 2026-05-05T19:23:26+00:00 | 2026-05-05T19:24:26+00:00 | 00:01:00 |
| train-token:random_forest | 2026-05-05T19:24:26+00:00 | 2026-05-05T19:43:32+00:00 | 00:19:06 |
| train-sentence:random_forest | 2026-05-05T19:43:32+00:00 | 2026-05-05T19:44:02+00:00 | 00:00:30 |
| train-heads:random_forest | 2026-05-05T19:44:02+00:00 | 2026-05-05T20:42:03+00:00 | 00:58:01 |
| train-token:mlp | 2026-05-05T20:42:03+00:00 | 2026-05-05T20:49:33+00:00 | 00:07:30 |
| train-sentence:mlp | 2026-05-05T20:49:33+00:00 | 2026-05-05T20:52:03+00:00 | 00:02:30 |
| train-heads:mlp | 2026-05-05T20:52:03+00:00 | 2026-05-05T21:25:33+00:00 | 00:33:30 |
| report | 2026-05-05T21:25:33+00:00 | 2026-05-05T21:25:34+00:00 | 00:00:01 |

The extraction stage dominated runtime at `26:21:53`. The slowest training
stage was random forest head training at `00:58:01`.

### Logs Created

The 50k run generated:

| log/state file | purpose |
|---|---|
| `artifacts/en_de_50k/run_state.json` | durable config, stages, events |
| `artifacts/en_de_50k/run_summary.json` | final metrics and top-head report |
| `artifacts/en_de_50k/environment.json` | Python/package/CUDA/GPU environment |
| `artifacts/en_de_50k/logs/RUN_LOG.md` | stage start/completion chronology |
| `artifacts/en_de_50k/logs/stage_status.md` | latest status table |
| `artifacts/en_de_50k/logs/heartbeat.jsonl` | periodic liveness records |
| `artifacts/en_de_50k/logs/*.stdout.log` | per-stage stdout |
| `artifacts/en_de_50k/logs/*.stderr.log` | per-stage stderr |
| `artifacts/en_de_50k/logs/CHANGELOG.md` | per-run local log initialization |
| `artifacts/en_de_50k/logs/run.lock` | active run lock while orchestrator is running |

`heartbeat.jsonl` contains 3,318 records. The first heartbeat was for `sample`
at `2026-05-04T16:59:33+00:00`; the final heartbeat was for
`train-heads:mlp` at `2026-05-05T21:25:33+00:00`.

The lock file was removed on clean completion. A missing lock plus completed
stage status means the run is not currently running and is complete.

### Recovery Behavior

The recovery design now covers the main local laptop failure modes:

| failure mode | recovery behavior |
|---|---|
| chat disconnects | PowerShell-launched local process can continue outside the chat |
| laptop shuts down during extraction | rerun skips completed chunks and retries missing/incomplete chunks |
| stale temp chunk output exists | chunk is not treated as complete unless final outputs and report exist |
| token/sentence training stops | stage reruns unless metrics, model, and test predictions are complete |
| sentence-head training stops | completed per-head work dirs are reused; missing/incomplete heads retrain |
| random forest all-head models would be huge | default `persist-head-models=best` stores only the selected best model |
| stale lock remains | status/orchestrator detects whether PID is live; stale locks are removed |
| CUDA unavailable for 50k | run stops instead of falling back to CPU |

The final 50k run did not record a recovery event in `run_state.json`; it has
only `run_start` and `run_complete` events. Recovery support was therefore
available but apparently not needed in the final successful 50k execution.

## 11. Current Branch Status

The current branch is `Mohit_dev`.

Recent commit history:

| commit | message |
|---|---|
| `697b558` | added recovery code for colab run |
| `50a97f3` | Merge remote-tracking branch `origin/main` into `Mohit_dev` |
| `2030dc4` | main pc |
| `34ef641` | Convert WMT22/WMT23 directories to tracked files |
| `454764d` | added a sentence (level, head) classifier |
| `f19b0ba` | changed the attention method to get the cross attention matrix |
| `1901c15` | Added WMT22 and 23 data |

At the time this document was prepared, there were uncommitted implementation
changes relative to the branch commit:

| status | path |
|---|---|
| modified | `pyproject.toml` |
| modified | `src/lucid_decoders/data/sample_ende.py` |
| modified | `src/lucid_decoders/io.py` |
| modified | `src/lucid_decoders/ml.py` |
| modified | `src/lucid_decoders/pipeline.py` |
| modified | `src/lucid_decoders/train/train_sentence_classifier.py` |
| modified | `src/lucid_decoders/train/train_sentence_head_classifier.py` |
| modified | `src/lucid_decoders/train/train_token_classifier.py` |
| untracked | `docs/` |
| untracked | `scripts/` |
| untracked | `src/lucid_decoders/tools/local_run.py` |
| untracked | `src/lucid_decoders/tools/local_status.py` |
| untracked | `tests/test_local_run_recovery.py` |
| untracked | `tests/test_sentence_head_recovery.py` |

The tracked diff for existing files was:

```text
8 files changed, 260 insertions(+), 33 deletions(-)
```

This does not count untracked new files such as the local recovery CLIs, tests,
scripts, and docs.

### Capability Difference From Main

Compared to `main`, this branch now has project capabilities that are not
present in the main baseline:

| capability | branch status |
|---|---|
| WMT22/WMT23 en-de trainable normalized data | available |
| token classifier | available |
| sentence classifier | available |
| sentence-head classifier | available |
| mBART attention feature extraction | available |
| chunked/resumable extraction | available |
| model comparisons | available |
| local laptop run orchestration | available |
| status and resume CLI | available |
| per-run persistent logs | available |
| per-head training recovery | available |
| `persist-head-models=best` | available |
| 50k completed local experiment | available in ignored artifacts |

The ignored generated artifacts should not be committed. The code, tests,
scripts, and tracked docs should be committed once reviewed.

## 12. Known Limitations

### Token Localization

Token localization is not yet strong enough to treat as a reliable final
localizer. The best current 50k token model is MLP:

```text
precision=0.084082
recall=0.656688
F1=0.149076
ROC-AUC=0.596523
```

The model catches many positive tokens but flags too many negative tokens.
Because the positive token rate is only `0.068133` in test, improving precision
requires better ranking features, better calibration, better thresholds, or a
more task-specific token modeling approach.

### Small Test Positive Ceiling

The original test split only provides 205 supervised positive examples. This
limits the statistical resolution of test metrics for 10k and 50k. The test
set is useful and stable, but a few examples can still move metrics.

### Threshold Sensitivity

The selected thresholds materially change precision/recall tradeoffs:

- random forest sentence at 50k uses threshold `0.25`, producing very high
  recall and many false positives
- MLP sentence at 50k uses threshold `0.05`, also favoring recall
- logistic sentence at 50k uses threshold `0.35`, giving the best balanced
  sentence F1 among sentence classifiers

Future reporting should include threshold curves and calibrated operating
points rather than one selected threshold.

### Compact Attention Summaries

The pipeline intentionally compresses raw attention matrices into summary
statistics. This makes scaling feasible, but it loses structure:

- exact source-token alignment patterns
- multi-source support patterns
- attention sharpness changes over neighboring target tokens
- phrase-level continuity
- raw head-specific attention shapes

The current results show the summaries are useful, especially at sentence/head
level, but token localization may need richer local features.

### Label Noise And Supervision Shape

The normalized data merges signals from WMT22/WMT23 style annotations. Sentence
labels and token/span labels may not be equally clean. Negative examples are
sentence-level negatives, not necessarily proof that every token is perfectly
good. Positive examples have token/span supervision where available, but spans
may not map perfectly onto mBART subword tokens.

### Scientific Claims

The current results are promising engineering and modeling evidence. They are
not final scientific conclusions. Before publication-style claims, the project
needs calibration, ablations, seed sensitivity, label audits, and a locked
experiment manifest.

## 13. Recommended Next Steps

### Immediate Engineering Steps

1. Validate 50k artifacts one more time with a read-only script:
   - feature parquet row counts
   - unique sentence `example_id` counts
   - `sentence_head_rows = sentence_rows * 192`
   - test prediction row counts
   - best-head prediction row counts
2. Review and commit the current implementation:
   - modified source files
   - new local recovery tools
   - new tests
   - launcher script
   - tracked documentation
3. Keep generated run outputs ignored:
   - `data/processed/en_de_50k`
   - `artifacts/en_de_50k`
   - previous run artifacts
4. Add a short reproducibility manifest for the 50k run:
   - git commit
   - command
   - environment
   - model name
   - sample seed and counts
   - artifact paths

### Token Localization Improvements

The next modeling work should focus on token localization before moving to a
much larger run:

1. Add class-imbalance-aware token training:
   - class weights
   - precision-oriented threshold tuning
   - validation PR-AUC
   - top-k token recall per sentence
2. Add token sequence context:
   - neighboring token features
   - previous/next token attention deltas
   - span-level aggregation
   - target subword boundary flags
3. Add lexical and alignment features:
   - source-target length ratio
   - token text category
   - punctuation/number/name indicators
   - approximate source lexical support
4. Evaluate localization at the span level, not only independent token rows.

### Sentence-Level Improvements

For sentence hallucination detection:

1. Keep logistic regression as the main baseline because it is strong,
   stable, and interpretable.
2. Add calibration curves and Brier score.
3. Report precision/recall at fixed operating points.
4. Run seed sensitivity on the 50k sampler.
5. Compare attention-only features against simple lexical/length baselines.

### Head Analysis

The head-ranking results deserve a focused analysis pass:

1. Inspect raw attention maps for L6 H13, L7 H3, L9 H13, L6 H9, and L8 H7.
2. Compare top heads on positive true positives, false positives, and false
   negatives.
3. Check whether high-ranked heads specialize in:
   - source grounding
   - target repetition
   - named entities
   - numbers
   - punctuation or sentence boundaries
4. Run ablations:
   - cross-only
   - self-only
   - ratio-only
   - last-layer-only
   - no length features
5. Verify whether the same heads stay strong across different random samples.

### Scaling Guidance

The pipeline is technically ready to scale beyond 50k, but the next scale step
should be chosen carefully.

Recommended path:

1. Do not jump straight to the full dataset until token diagnostics are better.
2. First run targeted 50k ablations that are cheaper than full extraction:
   - reuse existing 50k features
   - train feature subsets
   - tune thresholds
   - evaluate calibration
3. If token localization improves, run a 100k or full-dataset extraction using
   the same local recovery infrastructure.
4. Keep chunk size at `250` unless GPU memory evidence supports changing it.
5. Keep `persist-head-models=best` for random forest and MLP.
6. Preserve the 50k run as the current baseline for all future comparisons.

### Final Project State

The project has achieved the core requested pipeline:

- normalized WMT22/WMT23 en-de training examples
- supervised sampling with positive token/span labels
- real mBART attention extraction under teacher forcing
- token, sentence, and sentence-head feature tables
- token, sentence, and sentence-head classifiers
- model comparison across logistic regression, random forest, and MLP
- ranked layer/head outputs
- local laptop CUDA scaling to 50k examples
- recovery support for interrupted extraction and training
- durable logs and summaries

The main remaining gap is not pipeline completion. The main remaining gap is
model quality for token localization. Sentence-level and head-level results are
already strong enough to justify deeper analysis and ablations.
