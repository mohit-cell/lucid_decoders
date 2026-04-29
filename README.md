# lucid_decoders

Attention-based hallucination detection for neural machine translation with:

- `WMT22/WMT23` data ingestion
- `mBART` attention extraction
- token-level hallucination classification
- sentence-level hallucination classification
- qualitative attention heatmaps

## Project Layout

```text
src/lucid_decoders/
  data/          dataset loading and preprocessing
  models/        mBART attention extraction
  features/      token and sentence feature builders
  train/         baseline training entrypoints
  eval/          held-out evaluation entrypoints
  analysis/      qualitative heatmaps
tests/           small feature-pipeline tests
data/
  raw/           place WMT22/WMT23 files here
  processed/     normalized examples and features
artifacts/       saved models, metrics, and plots
```

## Setup

Create an environment and install the package:

```bash
pip install -e .
```

Core runtime dependencies:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `numpy`

## Pipeline

1. Normalize raw WMT files into a shared schema.
2. Split data into train/validation/test.
3. Run `mBART` with `output_attentions=True`.
4. Convert attention tensors into token-level features.
5. Aggregate token evidence into sentence-level features.
6. Train token and sentence classifiers.
7. Evaluate with ROC-AUC, F1, precision, and recall.
8. Inspect representative attention heatmaps.

## Example Commands

Normalize raw examples:

```bash
python -m lucid_decoders.data.preprocess \
  --input data/raw/wmt22/train.jsonl \
  --output data/processed/wmt22_normalized.jsonl \
  --source-col source \
  --target-col hypothesis \
  --sentence-label-col label \
  --language-pair en-de
```

Prepare the cloned WMT22/WMT23 `en-de` datasets:

```bash
python -m lucid_decoders.data.prepare_ende \
  --wmt22-root data/raw/wmt22 \
  --wmt23-root data/raw/wmt23 \
  --output-dir data/processed/en_de
```

Extract `mBART` features:

```bash
python -m lucid_decoders.models.mbart_attention \
  --input data/processed/wmt22_normalized.jsonl \
  --token-output data/processed/token_features.parquet \
  --sentence-output data/processed/sentence_features.parquet \
  --model-name facebook/mbart-large-50-many-to-many-mmt \
  --source-lang en_XX \
  --target-lang de_DE
```

Train token classifier:

```bash
python -m lucid_decoders.train.train_token_classifier \
  --features data/processed/token_features.parquet \
  --artifacts-dir artifacts/token_logreg \
  --model-type logistic_regression
```

Train sentence classifier:

```bash
python -m lucid_decoders.train.train_sentence_classifier \
  --features data/processed/sentence_features.parquet \
  --artifacts-dir artifacts/sentence_logreg \
  --model-type logistic_regression
```

## Notes

- `mBART` attention extraction is the expensive stage and should run on GPU.
- The baseline classifiers can usually run on CPU once features are cached.
- The WMT label format can differ between files, so the preprocessing CLI accepts explicit column mappings.
