"""Attention feature extraction for hallucination detection."""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any

EPSILON = 1e-12


def attention_entropy(distribution: list[float], *, normalized: bool = False) -> float:
    """Return Shannon entropy for one attention distribution."""

    probs = _normalize_distribution(distribution)
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    if normalized:
        return entropy / math.log(len(probs))
    return entropy


def normalize_attention_stack(attention: Any) -> list[list[float]]:
    """Average layers and heads into a target-token by source-token matrix.

    Accepts a Hugging Face cross-attention tuple shaped as
    ``layers x batch x heads x target x source``, a single layer, or an already
    averaged matrix.
    """

    data = _as_plain_data(attention)
    depth = _depth(data)
    if depth == 5:
        matrices = [_matrix_from_layer(layer) for layer in data]
        return _average_matrices(matrices)
    if depth in (2, 3, 4):
        return _matrix_from_layer(data)
    raise ValueError(f"Unsupported attention shape depth: {depth}")


def compute_token_features(
    cross_attention: Any,
    *,
    decoder_self_attention: Any | None = None,
    source_tokens: list[str] | None = None,
    target_tokens: list[str] | None = None,
    coverage_threshold: float = 0.05,
) -> list[dict[str, float | str | int]]:
    """Compute per-target-token grounding features from attention matrices."""

    cross_matrix = normalize_attention_stack(cross_attention)
    self_matrix = normalize_attention_stack(decoder_self_attention) if decoder_self_attention is not None else None
    rows: list[dict[str, float | str | int]] = []

    for target_index, cross_row in enumerate(cross_matrix):
        probs = _normalize_distribution(cross_row)
        cross_max = max(probs) if probs else 0.0
        cross_variance = _variance(probs)
        entropy = attention_entropy(probs)
        entropy_norm = attention_entropy(probs, normalized=True)
        source_coverage = sum(1 for p in probs if p >= coverage_threshold) / len(probs) if probs else 0.0

        self_max = 0.0
        if self_matrix is not None and target_index < len(self_matrix):
            self_probs = _normalize_distribution(self_matrix[target_index])
            self_max = max(self_probs) if self_probs else 0.0
        self_cross_ratio = self_max / (cross_max + EPSILON)
        attention_risk = _bounded_mean(
            entropy_norm,
            1.0 - cross_max,
            source_coverage,
            min(self_cross_ratio / 5.0, 1.0),
        )

        rows.append(
            {
                "target_index": target_index,
                "target_token": _token_at(target_tokens, target_index),
                "source_token_count": len(source_tokens) if source_tokens else len(probs),
                "cross_entropy": entropy,
                "cross_entropy_norm": entropy_norm,
                "cross_max": cross_max,
                "cross_variance": cross_variance,
                "source_coverage": source_coverage,
                "self_max": self_max,
                "self_cross_max_ratio": self_cross_ratio,
                "attention_risk_score": attention_risk,
            }
        )
    return rows


def aggregate_sentence_features(
    token_features: list[dict[str, float | str | int]],
    *,
    source_len: int | None = None,
    target_len: int | None = None,
) -> dict[str, float]:
    """Aggregate token-level attention features into one classifier vector."""

    if target_len is None:
        target_len = len(token_features)
    if source_len is None:
        source_len = int(token_features[0]["source_token_count"]) if token_features else 0

    output: dict[str, float] = {
        "source_token_count": float(source_len),
        "target_token_count": float(target_len),
        "length_ratio": float(target_len) / float(source_len) if source_len else 0.0,
    }

    numeric_fields = (
        "cross_entropy",
        "cross_entropy_norm",
        "cross_max",
        "cross_variance",
        "source_coverage",
        "self_max",
        "self_cross_max_ratio",
        "attention_risk_score",
    )
    for field in numeric_fields:
        values = [float(row[field]) for row in token_features if field in row]
        output.update(_summary_stats(field, values))

    entropy_values = [float(row["cross_entropy_norm"]) for row in token_features if "cross_entropy_norm" in row]
    max_values = [float(row["cross_max"]) for row in token_features if "cross_max" in row]
    output["high_entropy_token_share"] = _share(entropy_values, lambda value: value >= 0.8)
    output["low_cross_max_token_share"] = _share(max_values, lambda value: value <= 0.2)
    return output


def extract_sentence_features(
    cross_attention: Any,
    *,
    decoder_self_attention: Any | None = None,
    source_tokens: list[str] | None = None,
    target_tokens: list[str] | None = None,
) -> dict[str, float]:
    """Convenience wrapper for token and sentence attention features."""

    token_rows = compute_token_features(
        cross_attention,
        decoder_self_attention=decoder_self_attention,
        source_tokens=source_tokens,
        target_tokens=target_tokens,
    )
    return aggregate_sentence_features(
        token_rows,
        source_len=len(source_tokens) if source_tokens else None,
        target_len=len(target_tokens) if target_tokens else None,
    )


def _summary_stats(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": mean(values),
        f"{prefix}_std": pstdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_min": min(values),
        f"{prefix}_max": max(values),
    }


def _share(values: list[float], predicate: Any) -> float:
    return sum(1 for value in values if predicate(value)) / len(values) if values else 0.0


def _bounded_mean(*values: float) -> float:
    clean = [min(max(float(value), 0.0), 1.0) for value in values]
    return sum(clean) / len(clean) if clean else 0.0


def _normalize_distribution(values: list[float]) -> list[float]:
    clean = [max(float(value), 0.0) for value in values]
    total = sum(clean)
    if total <= EPSILON:
        return [0.0 for _ in clean]
    return [value / total for value in clean]


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    center = mean(values)
    return sum((value - center) ** 2 for value in values) / len(values)


def _as_plain_data(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        return value.tolist()
    if isinstance(value, tuple):
        return [_as_plain_data(item) for item in value]
    if isinstance(value, list):
        return [_as_plain_data(item) for item in value]
    return value


def _depth(value: Any) -> int:
    if not isinstance(value, list):
        return 0
    if not value:
        return 1
    return 1 + _depth(value[0])


def _matrix_from_layer(layer: Any) -> list[list[float]]:
    depth = _depth(layer)
    if depth == 4:
        if not layer:
            return []
        return _matrix_from_layer(layer[0])
    if depth == 3:
        return _average_matrices([_coerce_matrix(head) for head in layer])
    if depth == 2:
        return _coerce_matrix(layer)
    raise ValueError(f"Unsupported attention layer depth: {depth}")


def _coerce_matrix(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _average_matrices(matrices: list[list[list[float]]]) -> list[list[float]]:
    valid = [matrix for matrix in matrices if matrix]
    if not valid:
        return []
    target_len = len(valid[0])
    source_len = len(valid[0][0]) if target_len else 0
    totals = [[0.0 for _ in range(source_len)] for _ in range(target_len)]
    for matrix in valid:
        if len(matrix) != target_len or any(len(row) != source_len for row in matrix):
            raise ValueError("Attention matrices must share the same target/source shape.")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                totals[i][j] += value
    scale = 1.0 / len(valid)
    return [[value * scale for value in row] for row in totals]


def _token_at(tokens: list[str] | None, index: int) -> str:
    if not tokens or index >= len(tokens):
        return ""
    return tokens[index]
