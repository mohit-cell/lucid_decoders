"""Classifier training and inference for attention-derived features."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evaluation import classification_metrics, select_threshold

NON_FEATURE_COLUMNS = {
    "example_id",
    "id",
    "label",
    "source",
    "generated",
    "source_lang",
    "target_lang",
}


@dataclass
class TrainedHallucinationClassifier:
    """A fitted classifier plus feature metadata and decision threshold."""

    model: Any
    feature_names: list[str]
    threshold: float = 0.5
    classifier_type: str = "logistic_regression"

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[float]:
        matrix = build_feature_matrix(rows, self.feature_names)
        if hasattr(self.model, "predict_proba"):
            return [float(value) for value in self.model.predict_proba(matrix)[:, 1]]
        if hasattr(self.model, "decision_function"):
            import math

            return [1.0 / (1.0 + math.exp(-float(value))) for value in self.model.decision_function(matrix)]
        raise RuntimeError("Classifier does not expose probability or decision scores.")

    def predict(self, rows: list[dict[str, Any]]) -> list[int]:
        return [1 if score >= self.threshold else 0 for score in self.predict_proba(rows)]


def fit_classifier(
    feature_rows: list[dict[str, Any]],
    labels: list[int],
    *,
    classifier_type: str = "logistic_regression",
    validation_rows: list[dict[str, Any]] | None = None,
    validation_labels: list[int] | None = None,
    threshold_metric: str = "f1",
    random_state: int = 13,
) -> tuple[TrainedHallucinationClassifier, dict[str, float]]:
    """Fit a hallucination classifier from sentence-level attention features."""

    if not feature_rows:
        raise ValueError("No feature rows provided.")
    model = build_classifier(classifier_type, random_state=random_state)
    feature_names = infer_feature_names(feature_rows)
    matrix = build_feature_matrix(feature_rows, feature_names)
    model.fit(matrix, labels)

    trained = TrainedHallucinationClassifier(
        model=model,
        feature_names=feature_names,
        classifier_type=classifier_type,
    )

    eval_rows = validation_rows or feature_rows
    eval_labels = validation_labels or labels
    probabilities = trained.predict_proba(eval_rows)
    if validation_rows and validation_labels:
        threshold, metrics = select_threshold(eval_labels, probabilities, metric=threshold_metric)
        trained.threshold = threshold
    else:
        metrics = classification_metrics(eval_labels, probabilities, threshold=trained.threshold)
    return trained, metrics


def build_classifier(classifier_type: str, *, random_state: int = 13) -> Any:
    """Construct a scikit-learn classifier."""

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install the `ml` extra to train classifiers.") from exc

    if classifier_type == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    if classifier_type == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state)
    if classifier_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if classifier_type == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state),
        )
    raise ValueError(f"Unsupported classifier_type: {classifier_type}")


def infer_feature_names(rows: list[dict[str, Any]]) -> list[str]:
    """Infer numeric classifier feature names from row dictionaries."""

    names: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in NON_FEATURE_COLUMNS:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            names.add(key)
    return sorted(names)


def build_feature_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> list[list[float]]:
    """Convert feature dictionaries into a numeric matrix."""

    matrix: list[list[float]] = []
    for row in rows:
        matrix.append([_to_float(row.get(name, 0.0)) for name in feature_names])
    return matrix


def load_feature_csv(path: str | Path) -> tuple[list[dict[str, Any]], list[int] | None]:
    """Load sentence-level features written by the extraction pipeline."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [int(float(row["label"])) for row in rows] if rows and "label" in rows[0] and rows[0]["label"] != "" else None
    return rows, labels


def save_model(classifier: TrainedHallucinationClassifier, path: str | Path) -> None:
    """Persist a fitted classifier with joblib."""

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install the `ml` extra to save trained classifiers.") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)


def load_model(path: str | Path) -> TrainedHallucinationClassifier:
    """Load a fitted classifier from disk."""

    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install the `ml` extra to load trained classifiers.") from exc
    return joblib.load(path)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

