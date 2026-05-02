from __future__ import annotations

from typing import Any

import numpy as np

from lucid_decoders.config import FeatureConfig
from lucid_decoders.features.token_features import distribution_metrics
from lucid_decoders.schemas import AttentionExtraction


def build_sentence_head_feature_rows(
    extraction: AttentionExtraction,
    feature_config: FeatureConfig | None = None,
) -> list[dict[str, Any]]:
    feature_config = feature_config or FeatureConfig()
    content_indices = [
        idx for idx, (start, end) in enumerate(extraction.target_offsets) if end > start
    ]
    if not content_indices:
        return []

    layers, heads, _, _ = extraction.cross_attentions.shape
    rows: list[dict[str, Any]] = []
    for layer_idx in range(layers):
        for head_idx in range(heads):
            cross_metrics = _collect_token_metrics(
                extraction.cross_attentions,
                layer_idx,
                head_idx,
                content_indices,
                feature_config,
                is_self_attention=False,
            )
            self_metrics = _collect_token_metrics(
                extraction.self_attentions,
                layer_idx,
                head_idx,
                content_indices,
                feature_config,
                is_self_attention=True,
            )

            row: dict[str, Any] = {
                "example_id": extraction.example_id,
                "language_pair": extraction.language_pair,
                "split": extraction.split,
                "source_text": extraction.source_text,
                "hypothesis_text": extraction.hypothesis_text,
                "sentence_label": extraction.sentence_label,
                "layer_id": layer_idx,
                "head_id": head_idx,
                "source_length_tokens": float(len(extraction.source_tokens)),
                "target_length_tokens": float(len(content_indices)),
            }
            row.update(_summarize_metric_values(cross_metrics, "cross"))
            row.update(_summarize_metric_values(self_metrics, "self"))
            row["self_to_cross_max_ratio_mean"] = row["self_max_mean"] / max(
                row["cross_max_mean"],
                feature_config.epsilon,
            )
            row["self_to_cross_entropy_ratio_mean"] = row["self_entropy_mean"] / max(
                row["cross_entropy_mean"],
                feature_config.epsilon,
            )
            rows.append(row)
    return rows


def _collect_token_metrics(
    attention: np.ndarray,
    layer_idx: int,
    head_idx: int,
    content_indices: list[int],
    feature_config: FeatureConfig,
    is_self_attention: bool,
) -> dict[str, list[float]]:
    metric_values: dict[str, list[float]] = {
        "entropy": [],
        "max": [],
        "variance": [],
        "topk_mass": [],
    }
    for token_idx in content_indices:
        values = attention[layer_idx, head_idx, token_idx]
        if is_self_attention:
            values = values[: token_idx + 1]
        stats = distribution_metrics(
            values,
            topk=feature_config.topk,
            epsilon=feature_config.epsilon,
        )
        for name, value in stats.items():
            metric_values[name].append(value)
    return metric_values


def _summarize_metric_values(
    metric_values: dict[str, list[float]],
    prefix: str,
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        array = np.asarray(values, dtype=float)
        summary[f"{prefix}_{metric_name}_mean"] = float(array.mean())
        summary[f"{prefix}_{metric_name}_std"] = float(array.std())
        summary[f"{prefix}_{metric_name}_min"] = float(array.min())
        summary[f"{prefix}_{metric_name}_max"] = float(array.max())
    return summary
