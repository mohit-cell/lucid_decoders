from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lucid_decoders.features.contracts import (
    SENTENCE_HEAD_REQUIRED_COLUMNS,
    SENTENCE_REQUIRED_COLUMNS,
    TOKEN_REQUIRED_COLUMNS,
)
from lucid_decoders.train import (
    train_sentence_classifier,
    train_sentence_head_classifier,
    train_token_classifier,
)


class TrainingSmokeTests(unittest.TestCase):
    def test_token_sentence_and_head_training_write_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            token_features = root / "token_features.csv"
            sentence_features = root / "sentence_features.csv"
            sentence_head_features = root / "sentence_head_features.csv"
            build_feature_frame(TOKEN_REQUIRED_COLUMNS, "token").to_csv(token_features, index=False)
            build_feature_frame(SENTENCE_REQUIRED_COLUMNS, "sentence").to_csv(sentence_features, index=False)
            build_feature_frame(SENTENCE_HEAD_REQUIRED_COLUMNS, "sentence_head").to_csv(
                sentence_head_features,
                index=False,
            )

            run_main(
                train_token_classifier.main,
                [
                    "train_token_classifier",
                    "--features",
                    str(token_features),
                    "--artifacts-dir",
                    str(root / "token_artifacts"),
                ],
            )
            run_main(
                train_sentence_classifier.main,
                [
                    "train_sentence_classifier",
                    "--features",
                    str(sentence_features),
                    "--artifacts-dir",
                    str(root / "sentence_artifacts"),
                ],
            )
            run_main(
                train_sentence_head_classifier.main,
                [
                    "train_sentence_head_classifier",
                    "--features",
                    str(sentence_head_features),
                    "--artifacts-dir",
                    str(root / "head_artifacts"),
                    "--min-train-examples",
                    "2",
                ],
            )

            self.assertTrue((root / "token_artifacts" / "metrics.json").exists())
            self.assertTrue((root / "token_artifacts" / "test_predictions.parquet").exists())
            self.assertTrue((root / "sentence_artifacts" / "metrics.json").exists())
            self.assertTrue((root / "sentence_artifacts" / "test_predictions.parquet").exists())
            self.assertTrue((root / "head_artifacts" / "metrics.json").exists())
            self.assertTrue((root / "head_artifacts" / "head_metrics.csv").exists())


def build_feature_frame(required_columns: set[str], kind: str) -> pd.DataFrame:
    rows = []
    split_labels = [
        ("train", 0),
        ("train", 1),
        ("validation", 0),
        ("validation", 1),
        ("test", 0),
        ("test", 1),
    ]
    for idx, (split, label) in enumerate(split_labels):
        row = {}
        for column in required_columns:
            row[column] = value_for_column(column, kind, idx, split, label)
        rows.append(row)
    return pd.DataFrame(rows)


def value_for_column(column: str, kind: str, idx: int, split: str, label: int) -> object:
    if column == "example_id":
        return f"ex-{idx}"
    if column == "language_pair":
        return "en-de"
    if column == "split":
        return split
    if column == "source_text":
        return f"source {idx}"
    if column == "hypothesis_text":
        return f"hypothesis {idx}"
    if column == "token_text":
        return f"tok{idx}"
    if column == "sentence_label":
        return label
    if column == "token_label":
        return label
    if column == "layer_id":
        return 0
    if column == "head_id":
        return 0
    if column == "token_index":
        return idx
    if column in {"token_start_char", "token_end_char", "num_target_tokens"}:
        return idx + 1
    base = 0.1 * (idx + 1)
    if kind == "sentence_head":
        base += 0.05
    return base + (len(column) % 7) * 0.01


def run_main(main_func, argv: list[str]) -> None:
    with patch.object(sys, "argv", argv):
        main_func()


if __name__ == "__main__":
    unittest.main()
