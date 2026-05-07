from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lucid_decoders.features.contracts import SENTENCE_HEAD_REQUIRED_COLUMNS
from lucid_decoders.train import train_sentence_head_classifier


class SentenceHeadRecoveryTests(unittest.TestCase):
    def test_interrupted_head_training_resumes_completed_heads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features = root / "sentence_head_features.csv"
            artifacts = root / "head_artifacts"
            build_sentence_head_frame().to_csv(features, index=False)

            with self.assertRaises(RuntimeError):
                run_main(
                    [
                        "train_sentence_head_classifier",
                        "--features",
                        str(features),
                        "--artifacts-dir",
                        str(artifacts),
                        "--min-train-examples",
                        "2",
                        "--resume",
                        "--stop-after-heads",
                        "1",
                    ]
                )

            first_result = artifacts / "head_work" / "layer_00" / "head_00" / "result.json"
            self.assertTrue(first_result.exists())
            stale_temp = artifacts / "head_work" / "layer_00" / "head_01" / "test_predictions.tmp.parquet"
            stale_temp.parent.mkdir(parents=True, exist_ok=True)
            stale_temp.write_text("partial", encoding="utf-8")

            run_main(
                [
                    "train_sentence_head_classifier",
                    "--features",
                    str(features),
                    "--artifacts-dir",
                    str(artifacts),
                    "--min-train-examples",
                    "2",
                    "--resume",
                ]
            )

            self.assertFalse(stale_temp.exists())
            metrics = pd.read_csv(artifacts / "head_metrics.csv")
            predictions = pd.read_parquet(artifacts / "test_predictions.parquet")
            self.assertEqual(len(metrics), 2)
            self.assertEqual(predictions[["example_id", "layer_id", "head_id"]].drop_duplicates().shape[0], 4)
            self.assertTrue((artifacts / "best_model.pkl").exists())
            self.assertFalse((artifacts / "models_by_head.pkl").exists())


def build_sentence_head_frame() -> pd.DataFrame:
    rows = []
    for head_id in (0, 1):
        for idx, (split, label) in enumerate(
            [
                ("train", 0),
                ("train", 1),
                ("validation", 0),
                ("validation", 1),
                ("test", 0),
                ("test", 1),
            ]
        ):
            row = {}
            for column in SENTENCE_HEAD_REQUIRED_COLUMNS:
                row[column] = value_for_column(column, idx, split, label, head_id)
            rows.append(row)
    return pd.DataFrame(rows)


def value_for_column(column: str, idx: int, split: str, label: int, head_id: int) -> object:
    if column == "example_id":
        return f"{split}-{label}-h{head_id}"
    if column == "language_pair":
        return "en-de"
    if column == "split":
        return split
    if column == "source_text":
        return f"source {idx}"
    if column == "hypothesis_text":
        return f"hypothesis {idx}"
    if column == "sentence_label":
        return label
    if column == "layer_id":
        return 0
    if column == "head_id":
        return head_id
    if column in {"source_length_tokens", "target_length_tokens"}:
        return idx + 2
    return float(label) + head_id * 0.1 + (len(column) % 5) * 0.01


def run_main(argv: list[str]) -> None:
    with patch.object(sys, "argv", argv):
        train_sentence_head_classifier.main()


if __name__ == "__main__":
    unittest.main()
