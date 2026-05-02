from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_cross_attention_heatmap(
    attention: np.ndarray,
    source_tokens: list[str],
    target_tokens: list[str],
    output_path: str | Path,
    title: str | None = None,
) -> None:
    plt.figure(figsize=(max(8, len(source_tokens) * 0.45), max(4, len(target_tokens) * 0.45)))
    sns.heatmap(attention, cmap="viridis", xticklabels=source_tokens, yticklabels=target_tokens)
    plt.xlabel("Source tokens")
    plt.ylabel("Target tokens")
    if title:
        plt.title(title)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a cross-attention heatmap from an .npz file.")
    parser.add_argument("--input", required=True, help="Path to .npz with attention, source_tokens, target_tokens.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = np.load(args.input, allow_pickle=True)
    attention = payload["attention"]
    source_tokens = payload["source_tokens"].tolist()
    target_tokens = payload["target_tokens"].tolist()
    plot_cross_attention_heatmap(attention, source_tokens, target_tokens, args.output, args.title)


if __name__ == "__main__":
    main()

