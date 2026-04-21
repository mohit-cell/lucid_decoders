"""Evaluation helpers for hallucination classification."""

from __future__ import annotations

import math
from typing import Iterable


def classification_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute ROC-AUC, F1, precision, and recall."""

    labels = [int(value) for value in y_true]
    scores = [float(value) for value in y_prob]
    predictions = [1 if score >= threshold else 0 for score in scores]

    tp = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 1)
    fp = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 1)
    fn = sum(1 for y, pred in zip(labels, predictions) if y == 1 and pred == 0)
    tn = sum(1 for y, pred in zip(labels, predictions) if y == 0 and pred == 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if labels else 0.0

    return {
        "roc_auc": roc_auc(labels, scores),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "threshold": threshold,
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
    }


def roc_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    """Compute binary ROC-AUC using average ranks for ties."""

    pairs = sorted((float(score), int(label)) for label, score in zip(y_true, y_score))
    positive_count = sum(label for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return math.nan

    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        end = index
        while end < len(pairs) and pairs[end][0] == score:
            end += 1
        average_rank = (rank + rank + (end - index) - 1) / 2.0
        positives_in_group = sum(label for _, label in pairs[index:end])
        rank_sum_positive += positives_in_group * average_rank
        rank += end - index
        index = end

    return (rank_sum_positive - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def select_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    *,
    metric: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Select a decision threshold on validation scores."""

    labels = [int(value) for value in y_true]
    scores = [float(value) for value in y_prob]
    if not scores:
        raise ValueError("Cannot select a threshold for empty scores.")

    candidates = sorted(set([0.0, 0.5, 1.0, *scores]))
    best_threshold = candidates[0]
    best_metrics = classification_metrics(labels, scores, threshold=best_threshold)
    best_value = best_metrics.get(metric)
    if best_value is None:
        raise ValueError(f"Unsupported threshold metric: {metric}")

    for threshold in candidates[1:]:
        current = classification_metrics(labels, scores, threshold=threshold)
        current_value = current[metric]
        if current_value > best_value:
            best_threshold = threshold
            best_metrics = current
            best_value = current_value
    return best_threshold, best_metrics

