# lucid_decoders

Attention-based hallucination detection for English-to-German machine translation.

The pipeline normalizes WMT22/WMT23 en-de quality-estimation data, runs mBART with
teacher-forced target translations and `output_attentions=True`, converts attention
tensors into compact features, and trains:

- a token-level hallucination/error localization classifier
- a sentence-level hallucination probability classifier
- one sentence-level classifier per decoder `(layer, head)` for attention-head ranking

## Setup

```bash
git submodule update --init --recursive
python -m pip install -e ".[dev]"
python -m pytest -q
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

## Kaggle 15k Runbook

Kaggle runtime storage is temporary, so package outputs under `/kaggle/working`
before using **Save Version**. After enabling a Kaggle GPU, clone and install:

```bash
%cd /kaggle/working
!git clone https://github.com/mohit-cell/lucid_decoders.git
%cd /kaggle/working/lucid_decoders
!git checkout Mohit_dev
!pip install -e .
```

Print the exact notebook cells for the 15k balanced extraction/training run:

```bash
!lucid-kaggle-15k --mode commands
```

Check progress or finished artifacts inside Kaggle:

```bash
!lucid-kaggle-15k --mode status
```

Print recovery commands after a Kaggle reset:

```bash
!lucid-kaggle-15k --mode recovery
```

The generated runbook prepares WMT22/WMT23, samples a 15k balanced subset,
runs one direct chunked mBART extraction over 15k examples, trains
logistic-regression token/sentence/head models first, and packages outputs into
`/kaggle/working/lucid_decoders_kaggle_outputs`.

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
