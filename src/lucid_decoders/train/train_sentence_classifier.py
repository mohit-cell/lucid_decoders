from __future__ import annotations

import argparse
from pathlib import Path

from lucid_decoders.features.contracts import validate_feature_frame
from lucid_decoders.io import read_table, write_table
from lucid_decoders.ml import (
    binary_classification_metrics,
    build_estimator,
    get_default_feature_columns,
    predict_positive_proba,
    save_json,
    save_pickle,
    tune_threshold,
    validate_training_frame,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a sentence-level hallucination classifier.")
    parser.add_argument("--features", required=True, help="Sentence feature CSV, JSONL, or Parquet file.")
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "mlp"],
    )
    parser.add_argument("--label-col", default="sentence_label")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threshold", type=float)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame = read_table(args.features)
    validate_feature_frame(frame, "sentence")
    frame = frame[frame[args.label_col].notna()].copy()
    if "split" not in frame.columns:
        raise ValueError("Expected a `split` column in the feature table.")

    train_frame = frame[frame["split"] == "train"].copy()
    val_frame = frame[frame["split"] == "validation"].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    validate_training_frame(train_frame, args.label_col, "sentence classifier")
    feature_cols = get_default_feature_columns(frame, label_col=args.label_col)

    model = build_estimator(args.model_type, random_state=args.seed)
    model.fit(train_frame[feature_cols], train_frame[args.label_col].astype(int))

    val_probs = predict_positive_proba(model, val_frame, feature_cols)
    if args.threshold is not None:
        threshold = args.threshold
    elif val_frame.empty:
        threshold = 0.5
    else:
        threshold = tune_threshold(val_frame[args.label_col].astype(int), val_probs)
    val_metrics = binary_classification_metrics(
        val_frame[args.label_col].astype(int),
        val_probs,
        threshold=threshold,
    )

    test_probs = predict_positive_proba(model, test_frame, feature_cols)
    test_metrics = binary_classification_metrics(
        test_frame[args.label_col].astype(int),
        test_probs,
        threshold=threshold,
    )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(model, artifacts_dir / "model.pkl")
    save_json(
        {
            "feature_columns": feature_cols,
            "model_type": args.model_type,
            "label_col": args.label_col,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        },
        artifacts_dir / "metrics.json",
    )

    prediction_frame = test_frame[["example_id", args.label_col]].copy()
    prediction_frame["sentence_score"] = test_probs
    prediction_frame["sentence_pred"] = (test_probs >= threshold).astype(int)
    write_table(prediction_frame, artifacts_dir / "test_predictions.parquet")


if __name__ == "__main__":
    main()

