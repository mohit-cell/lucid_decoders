import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucid_decoders.features import (  # noqa: E402
    aggregate_sentence_features,
    attention_entropy,
    compute_token_features,
    normalize_attention_stack,
)


class FeatureExtractionTests(unittest.TestCase):
    def test_attention_entropy_uniform_distribution(self):
        self.assertAlmostEqual(attention_entropy([1.0, 1.0, 1.0]), math.log(3), places=6)
        self.assertAlmostEqual(attention_entropy([1.0, 1.0, 1.0], normalized=True), 1.0, places=6)

    def test_attention_stack_averages_layers_and_heads(self):
        attention = [
            [
                [
                    [[0.9, 0.1], [0.2, 0.8]],
                    [[0.7, 0.3], [0.4, 0.6]],
                ]
            ],
            [
                [
                    [[0.5, 0.5], [0.6, 0.4]],
                    [[0.3, 0.7], [0.8, 0.2]],
                ]
            ],
        ]

        matrix = normalize_attention_stack(attention)

        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 2)
        self.assertAlmostEqual(matrix[0][0], 0.6)
        self.assertAlmostEqual(matrix[1][1], 0.5)

    def test_token_and_sentence_features(self):
        cross_attention = [[0.9, 0.1, 0.0], [0.34, 0.33, 0.33]]
        decoder_attention = [[1.0, 0.0], [0.4, 0.6]]

        token_features = compute_token_features(
            cross_attention,
            decoder_self_attention=decoder_attention,
            source_tokens=["The", "cat", "</s>"],
            target_tokens=["El", "gato"],
        )
        sentence_features = aggregate_sentence_features(token_features)

        self.assertEqual(len(token_features), 2)
        self.assertGreater(token_features[1]["cross_entropy_norm"], token_features[0]["cross_entropy_norm"])
        self.assertGreaterEqual(token_features[1]["attention_risk_score"], token_features[0]["attention_risk_score"])
        self.assertIn("cross_entropy_norm_mean", sentence_features)
        self.assertIn("attention_risk_score_mean", sentence_features)
        self.assertEqual(sentence_features["target_token_count"], 2.0)
        self.assertEqual(sentence_features["source_token_count"], 3.0)


if __name__ == "__main__":
    unittest.main()
