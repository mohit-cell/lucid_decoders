from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lucid_decoders.features.contracts import validate_feature_frame
from lucid_decoders.io import read_table, write_table
from lucid_decoders.ml import (
    binary_classification_metrics,
    build_estimator,
    empty_binary_classification_metrics,
    get_default_feature_columns,
    predict_positive_proba,
    save_json,
    save_pickle,
    tune_threshold,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sentence-level hallucination classifiers for each decoder layer/head."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Sentence-head feature CSV, JSONL, or Parquet file.",
    )
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "mlp"],
    )
    parser.add_argument("--label-col", default="sentence_label")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--min-train-examples",
        type=int,
        default=20,
        help="Skip heads with fewer labeled training examples.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of layer/head classifiers to train concurrently.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame = read_table(args.features)
    validate_feature_frame(frame, "sentence_head")
    frame = frame[frame[args.label_col].notna()].copy()
    required_cols = {"split", "layer_id", "head_id", args.label_col}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required sentence-head columns: {sorted(missing)}")

    feature_cols = get_default_feature_columns(
        frame,
        label_col=args.label_col,
        exclude_cols={"layer_id", "head_id"},
    )
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    models: dict[tuple[int, int], Any] = {}
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    groups = list(frame.groupby(["layer_id", "head_id"], sort=True))
    if args.n_jobs == 1:
        results = [
            train_one_head(
                layer_id=layer_id,
                head_id=head_id,
                group=group,
                feature_cols=feature_cols,
                model_type=args.model_type,
                label_col=args.label_col,
                seed=args.seed,
                threshold_arg=args.threshold,
                min_train_examples=args.min_train_examples,
            )
            for (layer_id, head_id), group in groups
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(train_one_head)(
                layer_id=layer_id,
                head_id=head_id,
                group=group,
                feature_cols=feature_cols,
                model_type=args.model_type,
                label_col=args.label_col,
                seed=args.seed,
                threshold_arg=args.threshold,
                min_train_examples=args.min_train_examples,
            )
            for (layer_id, head_id), group in groups
        )

    for result in results:
        if result is None:
            continue
        metric_rows.append(result["metrics"])
        models[result["head_key"]] = result["model"]
        if result["predictions"] is not None:
            prediction_frames.append(result["predictions"])

    if not metric_rows:
        raise ValueError(
            "No head classifiers were trained. Check that the feature file has train rows "
            "with both positive and negative sentence labels."
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["rank_score"] = metrics_frame.apply(rank_score, axis=1)
    metrics_frame = metrics_frame.sort_values(
        by=["rank_score", "validation_f1", "test_f1"],
        ascending=False,
        na_position="last",
    )

    save_pickle(models, artifacts_dir / "models_by_head.pkl")
    write_table(metrics_frame, artifacts_dir / "head_metrics.csv")
    if prediction_frames:
        write_table(pd.concat(prediction_frames, ignore_index=True), artifacts_dir / "test_predictions.parquet")
    save_json(
        {
            "feature_columns": feature_cols,
            "model_type": args.model_type,
            "label_col": args.label_col,
            "num_head_classifiers": len(models),
            "best_head": {
                "layer_id": int(metrics_frame.iloc[0]["layer_id"]),
                "head_id": int(metrics_frame.iloc[0]["head_id"]),
                "rank_score": none_if_nan(metrics_frame.iloc[0]["rank_score"]),
                "validation_roc_auc": none_if_nan(metrics_frame.iloc[0]["validation_roc_auc"]),
                "validation_f1": none_if_nan(metrics_frame.iloc[0]["validation_f1"]),
                "test_roc_auc": none_if_nan(metrics_frame.iloc[0]["test_roc_auc"]),
                "test_f1": none_if_nan(metrics_frame.iloc[0]["test_f1"]),
            },
        },
        artifacts_dir / "metrics.json",
    )


def train_one_head(
    *,
    layer_id: int,
    head_id: int,
    group: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    label_col: str,
    seed: int,
    threshold_arg: float | None,
    min_train_examples: int,
) -> dict[str, Any] | None:
    train_frame = group[group["split"] == "train"].copy()
    val_frame = group[group["split"] == "validation"].copy()
    test_frame = group[group["split"] == "test"].copy()
    if len(train_frame) < min_train_examples:
        return None
    if train_frame[label_col].nunique() < 2:
        return None

    model = build_estimator(model_type, random_state=seed)
    model.fit(train_frame[feature_cols], train_frame[label_col].astype(int))

    if not val_frame.empty:
        val_probs = predict_positive_proba(model, val_frame, feature_cols)
        threshold = threshold_arg if threshold_arg is not None else tune_threshold(
            val_frame[label_col].astype(int),
            val_probs,
        )
        val_metrics = binary_classification_metrics(
            val_frame[label_col].astype(int),
            val_probs,
            threshold=threshold,
        )
    else:
        threshold = threshold_arg if threshold_arg is not None else 0.5
        val_metrics = empty_binary_classification_metrics(threshold)

    if not test_frame.empty:
        test_probs = predict_positive_proba(model, test_frame, feature_cols)
        test_metrics = binary_classification_metrics(
            test_frame[label_col].astype(int),
            test_probs,
            threshold=threshold,
        )
        prediction_frame = test_frame[["example_id", label_col]].copy()
        prediction_frame["layer_id"] = int(layer_id)
        prediction_frame["head_id"] = int(head_id)
        prediction_frame["sentence_score"] = test_probs
        prediction_frame["sentence_pred"] = (test_probs >= threshold).astype(int)
    else:
        test_metrics = empty_binary_classification_metrics(threshold)
        prediction_frame = None

    train_positive_rate = float(train_frame[label_col].astype(int).mean())
    metrics = {
        "layer_id": int(layer_id),
        "head_id": int(head_id),
        "train_examples": int(len(train_frame)),
        "validation_examples": int(len(val_frame)),
        "test_examples": int(len(test_frame)),
        "train_positive_rate": train_positive_rate,
        "threshold": float(threshold),
        **prefix_metrics(val_metrics, "validation"),
        **prefix_metrics(test_metrics, "test"),
    }
    return {
        "head_key": (int(layer_id), int(head_id)),
        "metrics": metrics,
        "model": model,
        "predictions": prediction_frame,
    }

def prefix_metrics(metrics: dict[str, float | None], prefix: str) -> dict[str, float | None]:
    return {
        f"{prefix}_{name}": value
        for name, value in metrics.items()
        if name != "threshold"
    }


def rank_score(row: pd.Series) -> float:
    for column in ("validation_roc_auc", "validation_f1", "test_roc_auc", "test_f1"):
        value = row.get(column)
        if value is not None and not pd.isna(value):
            return float(value)
    return float("-inf")


def none_if_nan(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


if __name__ == "__main__":
    main()
