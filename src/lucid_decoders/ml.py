from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lucid_decoders.io import ensure_parent_dir

META_COLUMNS = {
    "example_id",
    "language_pair",
    "split",
    "source_text",
    "hypothesis_text",
    "token_text",
    "token_index",
    "token_start_char",
    "token_end_char",
    "token_label",
    "sentence_label",
    "layer_id",
    "head_id",
}


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for training and evaluation. Install the project "
            "dependencies with `pip install -e .` first."
        ) from exc


def get_default_feature_columns(
    frame: pd.DataFrame,
    label_col: str,
    exclude_cols: set[str] | None = None,
) -> list[str]:
    exclude = set(META_COLUMNS)
    exclude.add(label_col)
    if exclude_cols:
        exclude.update(exclude_cols)
    numeric_cols = [
        col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col]) and col not in exclude
    ]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for training.")
    return sorted(numeric_cols)


def build_estimator(model_type: str, random_state: int = 13) -> Any:
    require_sklearn()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if model_type == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_type == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_type == "mlp":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=500,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def predict_positive_proba(model: Any, frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if frame.empty:
        return np.asarray([], dtype=float)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame[feature_cols])[:, 1]
        return np.asarray(probs, dtype=float)
    predictions = model.decision_function(frame[feature_cols])
    predictions = np.asarray(predictions, dtype=float)
    return 1.0 / (1.0 + np.exp(-predictions))


def tune_threshold(y_true: pd.Series, probs: np.ndarray) -> float:
    require_sklearn()
    from sklearn.metrics import f1_score

    if len(y_true) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        preds = (probs >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def binary_classification_metrics(
    y_true: pd.Series,
    probs: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    require_sklearn()
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    if len(y_true) == 0:
        return empty_binary_classification_metrics(threshold)
    preds = (probs >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    unique_labels = set(int(label) for label in pd.Series(y_true).dropna().unique())
    if len(unique_labels) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
    else:
        metrics["roc_auc"] = None
    return metrics


def empty_binary_classification_metrics(threshold: float) -> dict[str, float | None]:
    return {
        "threshold": float(threshold),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "roc_auc": None,
    }


def validate_training_frame(frame: pd.DataFrame, label_col: str, context: str) -> None:
    if frame.empty:
        raise ValueError(f"No labeled training rows found for {context}.")
    labels = frame[label_col].dropna().astype(int)
    if labels.nunique() < 2:
        raise ValueError(
            f"Training rows for {context} must contain both positive and negative labels."
        )


def save_pickle(obj: Any, path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
