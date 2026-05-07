from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lucid_decoders.features.contracts import validate_feature_frame
from lucid_decoders.io import read_table, write_table_atomic
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
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-head training outputs.")
    parser.add_argument(
        "--work-dir",
        help="Directory for per-head recovery artifacts. Defaults to <artifacts-dir>/head_work.",
    )
    parser.add_argument(
        "--persist-head-models",
        choices=["best", "all", "none"],
        default="best",
        help="Persist only the best head model, every per-head model, or no head models.",
    )
    parser.add_argument("--stop-after-heads", type=int, help=argparse.SUPPRESS)
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
    work_dir = Path(args.work_dir) if args.work_dir else artifacts_dir / "head_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    estimator_n_jobs = 1 if args.model_type == "random_forest" and args.n_jobs != 1 else None

    groups = list(frame.groupby(["layer_id", "head_id"], sort=True))
    if args.n_jobs == 1:
        results = []
        completed_count = 0
        for (layer_id, head_id), group in groups:
            result = train_or_resume_head(
                layer_id=layer_id,
                head_id=head_id,
                group=group,
                feature_cols=feature_cols,
                model_type=args.model_type,
                label_col=args.label_col,
                seed=args.seed,
                threshold_arg=args.threshold,
                min_train_examples=args.min_train_examples,
                work_dir=work_dir,
                resume=args.resume,
                persist_model=args.persist_head_models == "all",
                estimator_n_jobs=estimator_n_jobs,
            )
            results.append(result)
            if result is not None:
                completed_count += 1
            if args.stop_after_heads is not None and completed_count >= args.stop_after_heads:
                raise RuntimeError(f"Stopped after {completed_count} completed head classifiers.")
    else:
        if args.stop_after_heads is not None:
            raise ValueError("--stop-after-heads is only supported with --n-jobs 1.")
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
            delayed(train_or_resume_head)(
                layer_id=layer_id,
                head_id=head_id,
                group=group,
                feature_cols=feature_cols,
                model_type=args.model_type,
                label_col=args.label_col,
                seed=args.seed,
                threshold_arg=args.threshold,
                min_train_examples=args.min_train_examples,
                work_dir=work_dir,
                resume=args.resume,
                persist_model=args.persist_head_models == "all",
                estimator_n_jobs=estimator_n_jobs,
            )
            for (layer_id, head_id), group in groups
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    model_manifest: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            continue
        metric_rows.append(result["metrics"])
        if result.get("predictions_path"):
            prediction_frames.append(pd.read_parquet(result["predictions_path"]))
        if result.get("model_path"):
            model_manifest.append(
                {
                    "layer_id": result["layer_id"],
                    "head_id": result["head_id"],
                    "model_path": result["model_path"],
                }
            )

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

    write_table_atomic(metrics_frame, artifacts_dir / "head_metrics.csv")
    if prediction_frames:
        write_table_atomic(
            pd.concat(prediction_frames, ignore_index=True),
            artifacts_dir / "test_predictions.parquet",
        )
    best_model_path = None
    if args.persist_head_models in {"best", "all"}:
        best_layer_id = int(metrics_frame.iloc[0]["layer_id"])
        best_head_id = int(metrics_frame.iloc[0]["head_id"])
        best_group = frame[(frame["layer_id"] == best_layer_id) & (frame["head_id"] == best_head_id)]
        best_model = fit_head_model(
            group=best_group,
            feature_cols=feature_cols,
            model_type=args.model_type,
            label_col=args.label_col,
            seed=args.seed,
            estimator_n_jobs=estimator_n_jobs,
        )
        best_model_path = artifacts_dir / "best_model.pkl"
        save_pickle(best_model, best_model_path)
        save_json(
            {
                "layer_id": best_layer_id,
                "head_id": best_head_id,
                "model_path": str(best_model_path),
            },
            artifacts_dir / "best_model_info.json",
        )
    if model_manifest:
        save_json({"models": model_manifest}, artifacts_dir / "head_model_manifest.json")
    save_json(
        {
            "feature_columns": feature_cols,
            "model_type": args.model_type,
            "label_col": args.label_col,
            "num_head_classifiers": int(len(metrics_frame)),
            "persist_head_models": args.persist_head_models,
            "head_work_dir": str(work_dir),
            "best_model_path": str(best_model_path) if best_model_path else None,
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


def train_or_resume_head(
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
    work_dir: Path,
    resume: bool,
    persist_model: bool,
    estimator_n_jobs: int | None,
) -> dict[str, Any] | None:
    head_dir = build_head_dir(work_dir, int(layer_id), int(head_id))
    result_path = head_dir / "result.json"
    if resume:
        completed = read_completed_head_result(result_path)
        if completed is not None:
            return completed

    cleanup_temp_files(head_dir)
    result = train_one_head(
        layer_id=int(layer_id),
        head_id=int(head_id),
        group=group,
        feature_cols=feature_cols,
        model_type=model_type,
        label_col=label_col,
        seed=seed,
        threshold_arg=threshold_arg,
        min_train_examples=min_train_examples,
        estimator_n_jobs=estimator_n_jobs,
    )
    if result is None:
        return None

    head_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = head_dir / "metrics.json"
    predictions_path = head_dir / "test_predictions.parquet"
    model_path = head_dir / "model.pkl" if persist_model else None

    save_json(result["metrics"], metrics_path)
    prediction_frame = result["predictions"]
    prediction_rows = 0
    if prediction_frame is not None:
        prediction_rows = int(len(prediction_frame))
        write_table_atomic(prediction_frame, predictions_path)
    if model_path is not None:
        save_pickle(result["model"], model_path)

    completed = {
        "status": "completed",
        "layer_id": int(layer_id),
        "head_id": int(head_id),
        "metrics": result["metrics"],
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path) if prediction_frame is not None else None,
        "prediction_rows": prediction_rows,
        "model_path": str(model_path) if model_path is not None else None,
    }
    save_json(completed, result_path)
    return completed


def build_head_dir(work_dir: Path, layer_id: int, head_id: int) -> Path:
    return work_dir / f"layer_{layer_id:02d}" / f"head_{head_id:02d}"


def read_completed_head_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("status") != "completed":
        return None
    predictions_path = payload.get("predictions_path")
    if predictions_path and not Path(predictions_path).exists():
        return None
    model_path = payload.get("model_path")
    if model_path and not Path(model_path).exists():
        return None
    return payload


def cleanup_temp_files(head_dir: Path) -> None:
    if not head_dir.exists():
        return
    for temp_path in head_dir.rglob("*.tmp*"):
        if temp_path.is_file():
            temp_path.unlink()


def fit_head_model(
    *,
    group: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    label_col: str,
    seed: int,
    estimator_n_jobs: int | None,
) -> Any:
    train_frame = group[group["split"] == "train"].copy()
    model = build_estimator(model_type, random_state=seed, n_jobs=estimator_n_jobs)
    model.fit(train_frame[feature_cols], train_frame[label_col].astype(int))
    return model


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
    estimator_n_jobs: int | None = None,
) -> dict[str, Any] | None:
    train_frame = group[group["split"] == "train"].copy()
    val_frame = group[group["split"] == "validation"].copy()
    test_frame = group[group["split"] == "test"].copy()
    if len(train_frame) < min_train_examples:
        return None
    if train_frame[label_col].nunique() < 2:
        return None

    model = build_estimator(model_type, random_state=seed, n_jobs=estimator_n_jobs)
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
