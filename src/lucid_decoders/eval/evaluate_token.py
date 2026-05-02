from __future__ import annotations

import argparse
from pathlib import Path

from lucid_decoders.io import read_table
from lucid_decoders.ml import (
    binary_classification_metrics,
    load_pickle,
    predict_positive_proba,
    save_json,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained token classifier.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-columns", required=True, help="Path to training metrics.json.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-col", default="token_label")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--split", default="test")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    frame = read_table(args.features)
    frame = frame[frame[args.label_col].notna()].copy()
    if "split" in frame.columns:
        frame = frame[frame["split"] == args.split].copy()

    metadata = read_table_or_json(args.feature_columns)
    feature_columns = metadata["feature_columns"]
    model = load_pickle(args.model)
    probs = predict_positive_proba(model, frame, feature_columns)
    metrics = binary_classification_metrics(frame[args.label_col].astype(int), probs, args.threshold)
    save_json(metrics, args.output)


def read_table_or_json(path: str | Path) -> dict:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    main()

