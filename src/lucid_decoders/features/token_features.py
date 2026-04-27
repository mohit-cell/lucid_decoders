from __future__ import annotations

from typing import Any

import numpy as np

from lucid_decoders.config import FeatureConfig
from lucid_decoders.schemas import AttentionExtraction


def shannon_entropy(values: np.ndarray, epsilon: float = 1e-12) -> float:
    clipped = np.clip(values.astype(float), epsilon, 1.0)
    clipped = clipped / clipped.sum()
    return float(-(clipped * np.log(clipped)).sum())


def distribution_metrics(values: np.ndarray, topk: int, epsilon: float) -> dict[str, float]:
    normalized = np.clip(values.astype(float), epsilon, 1.0)
    normalized = normalized / normalized.sum()
    sorted_values = np.sort(normalized)[::-1]
    return {
        "entropy": shannon_entropy(normalized, epsilon),
        "max": float(normalized.max()),
        "variance": float(np.var(normalized)),
        "topk_mass": float(sorted_values[:topk].sum()),
    }


def summarize_attention_block(
    block: np.ndarray,
    prefix: str,
    feature_config: FeatureConfig,
) -> dict[str, float]:
    layers, heads, _ = block.shape
    metric_grids = {
        "entropy": np.zeros((layers, heads), dtype=float),
        "max": np.zeros((layers, heads), dtype=float),
        "variance": np.zeros((layers, heads), dtype=float),
        "topk_mass": np.zeros((layers, heads), dtype=float),
    }

    for layer_idx in range(layers):
        for head_idx in range(heads):
            stats = distribution_metrics(
                block[layer_idx, head_idx],
                topk=feature_config.topk,
                epsilon=feature_config.epsilon,
            )
            for name, value in stats.items():
                metric_grids[name][layer_idx, head_idx] = value

    summary: dict[str, float] = {}
    for name, grid in metric_grids.items():
        working_grid = grid[-1:, :] if feature_config.include_last_layer_only else grid
        summary[f"{prefix}_{name}_mean"] = float(working_grid.mean())
        summary[f"{prefix}_{name}_std"] = float(working_grid.std())
        summary[f"{prefix}_{name}_min"] = float(working_grid.min())
        summary[f"{prefix}_{name}_max"] = float(working_grid.max())
        summary[f"{prefix}_{name}_last_layer_mean"] = float(grid[-1].mean())
    return summary


def build_token_feature_rows(
    extraction: AttentionExtraction,
    feature_config: FeatureConfig | None = None,
) -> list[dict[str, Any]]:
    feature_config = feature_config or FeatureConfig()
    content_indices = [
        idx for idx, (start, end) in enumerate(extraction.target_offsets) if end > start
    ]
    token_rows: list[dict[str, Any]] = []
    total_content = max(len(content_indices), 1)

    for content_rank, token_idx in enumerate(content_indices):
        cross_block = extraction.cross_attentions[:, :, token_idx, :]
        self_block = extraction.self_attentions[:, :, token_idx, : token_idx + 1]
        cross_features = summarize_attention_block(cross_block, "cross", feature_config)
        self_features = summarize_attention_block(self_block, "self", feature_config)
        token_start, token_end = extraction.target_offsets[token_idx]

        row: dict[str, Any] = {
            "example_id": extraction.example_id,
            "language_pair": extraction.language_pair,
            "split": extraction.split,
            "source_text": extraction.source_text,
            "hypothesis_text": extraction.hypothesis_text,
            "sentence_label": extraction.sentence_label,
            "token_label": extraction.token_labels[token_idx] if extraction.token_labels is not None else None,
            "token_index": token_idx,
            "token_text": extraction.target_tokens[token_idx],
            "token_start_char": token_start,
            "token_end_char": token_end,
            "token_relative_position": float(content_rank / total_content),
            "source_length_tokens": float(len(extraction.source_tokens)),
            "target_length_tokens": float(total_content),
        }
        row.update(cross_features)
        row.update(self_features)
        row["self_to_cross_max_ratio"] = row["self_max_mean"] / max(
            row["cross_max_mean"],
            feature_config.epsilon,
        )
        row["self_to_cross_entropy_ratio"] = row["self_entropy_mean"] / max(
            row["cross_entropy_mean"],
            feature_config.epsilon,
        )
        token_rows.append(row)
    return token_rows

