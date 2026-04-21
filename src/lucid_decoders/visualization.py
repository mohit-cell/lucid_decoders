"""Attention heatmap visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .features import normalize_attention_stack


def plot_attention_heatmap(
    cross_attention: Any,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str | Path,
    *,
    title: str = "mBART Cross-Attention",
    max_source_tokens: int = 80,
    max_target_tokens: int = 80,
) -> Path:
    """Save a target-by-source cross-attention heatmap."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install the `ml` extra to generate attention heatmaps.") from exc

    matrix = normalize_attention_stack(cross_attention)
    matrix = [row[:max_source_tokens] for row in matrix[:max_target_tokens]]
    source_labels = source_tokens[:max_source_tokens]
    target_labels = target_tokens[:max_target_tokens]

    width = max(8.0, min(24.0, len(source_labels) * 0.35))
    height = max(6.0, min(24.0, len(target_labels) * 0.28))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Source tokens")
    ax.set_ylabel("Generated translation tokens")
    ax.set_xticks(range(len(source_labels)))
    ax.set_xticklabels(source_labels, rotation=70, ha="right", fontsize=8)
    ax.set_yticks(range(len(target_labels)))
    ax.set_yticklabels(target_labels, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output

