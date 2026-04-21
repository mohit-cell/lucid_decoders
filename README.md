# Lucid Decoders

Attention-based hallucination detection for Neural Machine Translation (NMT)
using mBART cross-attention.

The project implements the planned pipeline:

- input: source sentence plus generated translation
- model: `facebook/mbart-large-50-many-to-many-mmt`
- features: cross-attention entropy, max attention, variance, coverage, and decoder self-attention ratio
- classifier: logistic regression baseline, random forest, gradient boosting, or a small MLP
- output: hallucination probability in `[0, 1]`
- analysis: token-level feature tables and attention heatmaps

## Setup

Create a Python environment and install the ML dependencies:

```bash
pip install -e ".[ml]"
```

The first mBART run downloads model weights from Hugging Face, so use a machine
with enough disk space and memory.

## Data Format

Use a CSV, JSON, or JSONL file with at least:

```csv
source,generated,label
"The cat eats fish.","Die Katze isst Fisch.",0
"The cat eats fish.","Der Hund gewann die Wahl.",1
```

Recommended columns:

- `source`: source sentence/context
- `generated`: generated machine translation
- `label`: binary hallucination label, where `1` means hallucinated and `0` means faithful
- `source_lang`: optional mBART source language code, such as `en_XX`
- `target_lang`: optional mBART target language code, such as `de_DE`
- `token_labels`: optional token-level labels as a JSON list

WMT22/WMT23 hallucination data can be used after mapping its schema into these
columns, or by passing custom column names to the CLI.

## Feature Extraction

```bash
lucid-decoders extract-features \
  --data data/wmt_hallucination.csv \
  --output artifacts/features.csv \
  --source-lang en_XX \
  --target-lang de_DE
```

This writes:

- `artifacts/features.csv`: sentence-level classifier features
- `artifacts/features.tokens.csv`: token-level attention features for qualitative analysis

For a Hugging Face dataset mirror:

```bash
lucid-decoders extract-features \
  --hf-dataset DATASET_NAME \
  --split train \
  --source-col source \
  --generated-col generated \
  --label-col label \
  --output artifacts/features.csv \
  --source-lang en_XX \
  --target-lang de_DE
```

## Training

Train the interpretable logistic regression baseline:

```bash
lucid-decoders train \
  --features artifacts/features.csv \
  --model-output artifacts/logreg.joblib \
  --classifier logistic_regression
```

Alternative classifiers:

```bash
--classifier random_forest
--classifier gradient_boosting
--classifier mlp
```

If you have a validation split, use it for threshold tuning:

```bash
lucid-decoders train \
  --features artifacts/train_features.csv \
  --validation-features artifacts/validation_features.csv \
  --model-output artifacts/logreg.joblib
```

## Evaluation and Prediction

Evaluate ROC-AUC, F1, precision, and recall:

```bash
lucid-decoders evaluate \
  --features artifacts/test_features.csv \
  --model artifacts/logreg.joblib
```

If token labels are available and were exported to `*.tokens.csv`, evaluate the
token-level attention risk score:

```bash
lucid-decoders evaluate-tokens \
  --token-features artifacts/test_features.tokens.csv \
  --label-col token_label \
  --score-col attention_risk_score
```

Score unlabeled examples after extracting features:

```bash
lucid-decoders predict \
  --features artifacts/unlabeled_features.csv \
  --model artifacts/logreg.joblib \
  --output artifacts/predictions.csv
```

## Interpretability

Generate a cross-attention heatmap for a single translation:

```bash
lucid-decoders heatmap \
  --source "The cat eats fish." \
  --generated "Die Katze isst Fisch." \
  --source-lang en_XX \
  --target-lang de_DE \
  --output artifacts/attention_heatmap.png
```

Use the heatmap and `*.tokens.csv` outputs to inspect whether suspicious
generated tokens have high cross-attention entropy, low max source attention, or
high decoder-self/cross-attention ratios.

## Development Checks

The unit tests cover attention feature math and evaluation metrics without
requiring mBART downloads:

```bash
python -m unittest discover -s tests
```
