from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lucid_decoders.features.sentence_head_features import build_sentence_head_feature_rows
from lucid_decoders.features.sentence_features import build_sentence_feature_frame
from lucid_decoders.features.token_features import build_token_feature_rows
from lucid_decoders.schemas import AttentionExtraction


class FeaturePipelineTests(unittest.TestCase):
    def test_token_feature_builder_filters_special_tokens(self) -> None:
        cross = np.array(
            [
                [
                    [
                        [0.40, 0.40, 0.20],
                        [0.80, 0.10, 0.10],
                        [0.25, 0.25, 0.50],
                        [0.33, 0.33, 0.34],
                    ],
                    [
                        [0.30, 0.50, 0.20],
                        [0.70, 0.20, 0.10],
                        [0.20, 0.30, 0.50],
                        [0.34, 0.33, 0.33],
                    ],
                ],
                [
                    [
                        [0.50, 0.30, 0.20],
                        [0.75, 0.15, 0.10],
                        [0.15, 0.30, 0.55],
                        [0.50, 0.25, 0.25],
                    ],
                    [
                        [0.45, 0.35, 0.20],
                        [0.68, 0.22, 0.10],
                        [0.18, 0.28, 0.54],
                        [0.40, 0.30, 0.30],
                    ],
                ],
            ],
            dtype=float,
        )
        self_attention = np.array(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.20, 0.80, 0.0, 0.0],
                        [0.10, 0.30, 0.60, 0.0],
                        [0.10, 0.20, 0.20, 0.50],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.30, 0.70, 0.0, 0.0],
                        [0.15, 0.35, 0.50, 0.0],
                        [0.20, 0.20, 0.25, 0.35],
                    ],
                ],
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.75, 0.0, 0.0],
                        [0.10, 0.20, 0.70, 0.0],
                        [0.10, 0.15, 0.25, 0.50],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.35, 0.65, 0.0, 0.0],
                        [0.12, 0.28, 0.60, 0.0],
                        [0.20, 0.15, 0.25, 0.40],
                    ],
                ],
            ],
            dtype=float,
        )

        extraction = AttentionExtraction(
            example_id="ex-1",
            source_text="John arrived on Monday.",
            hypothesis_text="John came Monday.",
            source_tokens=["John", "arrived", "Monday"],
            target_tokens=["<lang>", "John", "came", "</s>"],
            target_offsets=[(0, 0), (0, 4), (5, 9), (0, 0)],
            cross_attentions=cross,
            self_attentions=self_attention,
            sentence_label=1,
            token_labels=[0, 0, 1, 0],
            language_pair="en-de",
            split="train",
        )

        rows = build_token_feature_rows(extraction)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["token_text"], "John")
        self.assertEqual(rows[1]["token_text"], "came")
        self.assertEqual(rows[1]["token_label"], 1)
        self.assertIn("cross_entropy_mean", rows[0])
        self.assertIn("self_to_cross_max_ratio", rows[0])

    def test_sentence_feature_builder_aggregates_numeric_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "example_id": "ex-1",
                    "language_pair": "en-de",
                    "split": "train",
                    "source_text": "A",
                    "hypothesis_text": "B",
                    "sentence_label": 1,
                    "token_label": 0,
                    "token_index": 1,
                    "token_text": "foo",
                    "token_start_char": 0,
                    "token_end_char": 3,
                    "cross_entropy_mean": 0.2,
                    "self_entropy_mean": 0.4,
                },
                {
                    "example_id": "ex-1",
                    "language_pair": "en-de",
                    "split": "train",
                    "source_text": "A",
                    "hypothesis_text": "B",
                    "sentence_label": 1,
                    "token_label": 1,
                    "token_index": 2,
                    "token_text": "bar",
                    "token_start_char": 4,
                    "token_end_char": 7,
                    "cross_entropy_mean": 0.8,
                    "self_entropy_mean": 0.6,
                },
            ]
        )

        sentence_frame = build_sentence_feature_frame(frame)
        self.assertEqual(len(sentence_frame), 1)
        row = sentence_frame.iloc[0]
        self.assertAlmostEqual(row["cross_entropy_mean_mean"], 0.5)
        self.assertAlmostEqual(row["self_entropy_mean_max"], 0.6)
        self.assertAlmostEqual(row["token_positive_fraction"], 0.5)

    def test_sentence_head_feature_builder_preserves_layer_head_identity(self) -> None:
        cross = np.array(
            [
                [
                    [
                        [0.40, 0.40, 0.20],
                        [0.80, 0.10, 0.10],
                        [0.25, 0.25, 0.50],
                        [0.33, 0.33, 0.34],
                    ],
                    [
                        [0.30, 0.50, 0.20],
                        [0.70, 0.20, 0.10],
                        [0.20, 0.30, 0.50],
                        [0.34, 0.33, 0.33],
                    ],
                ],
                [
                    [
                        [0.50, 0.30, 0.20],
                        [0.75, 0.15, 0.10],
                        [0.15, 0.30, 0.55],
                        [0.50, 0.25, 0.25],
                    ],
                    [
                        [0.45, 0.35, 0.20],
                        [0.68, 0.22, 0.10],
                        [0.18, 0.28, 0.54],
                        [0.40, 0.30, 0.30],
                    ],
                ],
            ],
            dtype=float,
        )
        self_attention = np.array(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.20, 0.80, 0.0, 0.0],
                        [0.10, 0.30, 0.60, 0.0],
                        [0.10, 0.20, 0.20, 0.50],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.30, 0.70, 0.0, 0.0],
                        [0.15, 0.35, 0.50, 0.0],
                        [0.20, 0.20, 0.25, 0.35],
                    ],
                ],
                [
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.25, 0.75, 0.0, 0.0],
                        [0.10, 0.20, 0.70, 0.0],
                        [0.10, 0.15, 0.25, 0.50],
                    ],
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.35, 0.65, 0.0, 0.0],
                        [0.12, 0.28, 0.60, 0.0],
                        [0.20, 0.15, 0.25, 0.40],
                    ],
                ],
            ],
            dtype=float,
        )
        extraction = AttentionExtraction(
            example_id="ex-1",
            source_text="John arrived on Monday.",
            hypothesis_text="John came Monday.",
            source_tokens=["John", "arrived", "Monday"],
            target_tokens=["<lang>", "John", "came", "</s>"],
            target_offsets=[(0, 0), (0, 4), (5, 9), (0, 0)],
            cross_attentions=cross,
            self_attentions=self_attention,
            sentence_label=1,
            token_labels=[0, 0, 1, 0],
            language_pair="en-de",
            split="train",
        )

        rows = build_sentence_head_feature_rows(extraction)
        self.assertEqual(len(rows), 4)
        self.assertEqual({(row["layer_id"], row["head_id"]) for row in rows}, {(0, 0), (0, 1), (1, 0), (1, 1)})
        self.assertTrue(all(row["sentence_label"] == 1 for row in rows))
        self.assertIn("cross_entropy_mean", rows[0])
        self.assertIn("self_to_cross_entropy_ratio_mean", rows[0])


if __name__ == "__main__":
    unittest.main()
