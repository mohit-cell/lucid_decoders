"""Command-line interface for the hallucination detection pipeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .data import load_examples
from .evaluation import classification_metrics
from .features import compute_token_features, extract_sentence_features
from .modeling import fit_classifier, load_feature_csv, load_model, save_model


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="mBART attention-based hallucination detection")
    subparsers = parser.add_subparsers(required=True)

    extract = subparsers.add_parser("extract-features", help="extract mBART attention features")
    _add_data_args(extract)
    extract.add_argument("--output", required=True, help="Feature CSV output path")
    extract.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    extract.add_argument("--source-lang", default="en_XX")
    extract.add_argument("--target-lang", default="de_DE")
    extract.add_argument("--max-examples", type=int, default=None)
    extract.set_defaults(func=cmd_extract_features)

    train = subparsers.add_parser("train", help="train a hallucination classifier from feature CSV")
    train.add_argument("--features", required=True)
    train.add_argument("--validation-features")
    train.add_argument("--model-output", required=True)
    train.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=("logistic_regression", "random_forest", "gradient_boosting", "mlp"),
    )
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a trained classifier on feature CSV")
    evaluate.add_argument("--features", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.set_defaults(func=cmd_evaluate)

    evaluate_tokens = subparsers.add_parser(
        "evaluate-tokens",
        help="evaluate token-level attention risk scores when token labels are available",
    )
    evaluate_tokens.add_argument("--token-features", required=True)
    evaluate_tokens.add_argument("--label-col", default="token_label")
    evaluate_tokens.add_argument("--score-col", default="attention_risk_score")
    evaluate_tokens.add_argument("--threshold", type=float, default=0.5)
    evaluate_tokens.set_defaults(func=cmd_evaluate_tokens)

    predict = subparsers.add_parser("predict", help="score examples from feature CSV")
    predict.add_argument("--features", required=True)
    predict.add_argument("--model", required=True)
    predict.add_argument("--output", required=True)
    predict.set_defaults(func=cmd_predict)

    heatmap = subparsers.add_parser("heatmap", help="generate one mBART cross-attention heatmap")
    heatmap.add_argument("--source", required=True)
    heatmap.add_argument("--generated", required=True)
    heatmap.add_argument("--output", required=True)
    heatmap.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    heatmap.add_argument("--source-lang", default="en_XX")
    heatmap.add_argument("--target-lang", default="de_DE")
    heatmap.set_defaults(func=cmd_heatmap)

    return parser


def cmd_extract_features(args: argparse.Namespace) -> int:
    from .attention import MBartAttentionConfig, MBartAttentionExtractor

    examples = load_examples(
        path=args.data,
        hf_dataset=args.hf_dataset,
        split=args.split,
        source_col=args.source_col,
        generated_col=args.generated_col,
        label_col=args.label_col,
        id_col=args.id_col,
        source_lang_col=args.source_lang_col,
        target_lang_col=args.target_lang_col,
        token_labels_col=args.token_labels_col,
        default_source_lang=args.source_lang,
        default_target_lang=args.target_lang,
    )
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    extractor = MBartAttentionExtractor(
        MBartAttentionConfig(
            model_name=args.model_name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    )

    rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for example in examples:
        result = extractor.extract(
            example.source,
            example.generated,
            source_lang=example.source_lang,
            target_lang=example.target_lang,
        )
        sentence_features = extract_sentence_features(
            result.cross_attentions,
            decoder_self_attention=result.decoder_attentions,
            source_tokens=result.source_tokens,
            target_tokens=result.target_tokens,
        )
        row: dict[str, Any] = {
            "example_id": example.example_id,
            "label": "" if example.label is None else example.label,
            "source_lang": example.source_lang or args.source_lang,
            "target_lang": example.target_lang or args.target_lang,
            **sentence_features,
        }
        rows.append(row)

        for token_feature in compute_token_features(
            result.cross_attentions,
            decoder_self_attention=result.decoder_attentions,
            source_tokens=result.source_tokens,
            target_tokens=result.target_tokens,
        ):
            token_index = int(token_feature["target_index"])
            if example.token_labels is not None and token_index < len(example.token_labels):
                token_feature["token_label"] = example.token_labels[token_index]
            token_rows.append({"example_id": example.example_id, **token_feature})

    _write_csv(args.output, rows)
    token_output = _with_suffix(args.output, ".tokens.csv")
    _write_csv(token_output, token_rows)
    print(json.dumps({"features": args.output, "token_features": str(token_output), "examples": len(rows)}, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    rows, labels = load_feature_csv(args.features)
    if labels is None:
        raise ValueError("Training feature CSV must contain a label column.")

    validation_rows = None
    validation_labels = None
    if args.validation_features:
        validation_rows, validation_labels = load_feature_csv(args.validation_features)
        if validation_labels is None:
            raise ValueError("Validation feature CSV must contain a label column.")

    classifier, metrics = fit_classifier(
        rows,
        labels,
        classifier_type=args.classifier,
        validation_rows=validation_rows,
        validation_labels=validation_labels,
    )
    save_model(classifier, args.model_output)
    print(json.dumps({"model": args.model_output, "metrics": metrics}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    rows, labels = load_feature_csv(args.features)
    if labels is None:
        raise ValueError("Evaluation feature CSV must contain a label column.")
    classifier = load_model(args.model)
    probabilities = classifier.predict_proba(rows)
    metrics = classification_metrics(labels, probabilities, threshold=classifier.threshold)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    rows, _ = load_feature_csv(args.features)
    classifier = load_model(args.model)
    probabilities = classifier.predict_proba(rows)
    predictions = [1 if score >= classifier.threshold else 0 for score in probabilities]
    output_rows = []
    for row, probability, prediction in zip(rows, probabilities, predictions):
        output_rows.append(
            {
                "example_id": row.get("example_id", ""),
                "hallucination_probability": probability,
                "prediction": prediction,
            }
        )
    _write_csv(args.output, output_rows)
    print(json.dumps({"predictions": args.output, "examples": len(output_rows)}, indent=2))
    return 0


def cmd_evaluate_tokens(args: argparse.Namespace) -> int:
    with Path(args.token_features).open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get(args.label_col) not in (None, "") and row.get(args.score_col) not in (None, "")
        ]
    if not rows:
        raise ValueError("No token rows with both labels and scores were found.")

    labels = [int(float(row[args.label_col])) for row in rows]
    scores = [float(row[args.score_col]) for row in rows]
    metrics = classification_metrics(labels, scores, threshold=args.threshold)
    print(json.dumps(metrics, indent=2))
    return 0


def cmd_heatmap(args: argparse.Namespace) -> int:
    from .attention import MBartAttentionConfig, MBartAttentionExtractor
    from .visualization import plot_attention_heatmap

    extractor = MBartAttentionExtractor(
        MBartAttentionConfig(
            model_name=args.model_name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
        )
    )
    result = extractor.extract(args.source, args.generated)
    output = plot_attention_heatmap(
        result.cross_attentions,
        result.source_tokens,
        result.target_tokens,
        args.output,
    )
    print(json.dumps({"heatmap": str(output)}, indent=2))
    return 0


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Local CSV/JSON/JSONL dataset path")
    source.add_argument("--hf-dataset", help="Hugging Face dataset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source-col", default="source")
    parser.add_argument("--generated-col", default="generated")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--id-col")
    parser.add_argument("--source-lang-col")
    parser.add_argument("--target-lang-col")
    parser.add_argument("--token-labels-col")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _with_suffix(path: str | Path, suffix: str) -> Path:
    original = Path(path)
    return original.with_name(f"{original.stem}{suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
