import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lucid_decoders.evaluation import classification_metrics, roc_auc, select_threshold  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_roc_auc_perfect_and_reversed(self):
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)

    def test_roc_auc_single_class_is_nan(self):
        self.assertTrue(math.isnan(roc_auc([1, 1], [0.4, 0.6])))

    def test_classification_metrics(self):
        metrics = classification_metrics([0, 1, 1, 0], [0.1, 0.9, 0.4, 0.2], threshold=0.5)

        self.assertEqual(metrics["true_positive"], 1.0)
        self.assertEqual(metrics["false_negative"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_select_threshold(self):
        threshold, metrics = select_threshold([0, 1, 1, 0], [0.2, 0.45, 0.9, 0.8])

        self.assertIn(threshold, {0.0, 0.2, 0.45, 0.5, 0.8, 0.9, 1.0})
        self.assertGreaterEqual(metrics["f1"], 0.5)


if __name__ == "__main__":
    unittest.main()

