from __future__ import annotations

import unittest

from lucid_decoders.data.sample_ende import sample_balanced_examples


class SampleEnDeTests(unittest.TestCase):
    def test_sample_balanced_examples_requires_positive_token_supervision(self) -> None:
        records = []
        for split in ("train", "validation", "test"):
            for idx in range(3):
                records.append(build_record(split, 0, idx, False))
                records.append(build_record(split, 1, idx, idx > 0))

        selected, summary = sample_balanced_examples(
            records,
            counts_per_label={"train": 1, "validation": 1, "test": 1},
            seed=13,
        )

        self.assertEqual(len(selected), 6)
        self.assertEqual(summary["selected_examples"], 6)
        self.assertTrue(
            all(
                record["sentence_label"] == 0 or record["hallucination_spans"]
                for record in selected
            )
        )

    def test_sample_balanced_examples_rewrites_duplicate_example_ids(self) -> None:
        records = []
        for split in ("train", "validation", "test"):
            records.append(build_record(split, 0, 0, False, example_id=f"{split}-negative"))
            records.append(build_record(split, 1, 0, True, example_id=f"{split}-positive"))
        records[0]["example_id"] = "duplicate"
        records[2]["example_id"] = "duplicate"

        selected, summary = sample_balanced_examples(
            records,
            counts_per_label={"train": 1, "validation": 1, "test": 1},
            seed=13,
        )

        ids = [record["example_id"] for record in selected]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(summary["rewritten_duplicate_example_ids"], 2)


def build_record(
    split: str,
    label: int,
    idx: int,
    supervised: bool,
    example_id: str | None = None,
) -> dict:
    return {
        "example_id": example_id or f"{split}-{label}-{idx}",
        "source_text": "source",
        "hypothesis_text": "target",
        "sentence_label": label,
        "hallucination_spans": [{"start": 0, "end": 1, "label": 1}] if supervised else [],
        "token_labels": None,
        "language_pair": "en-de",
        "split": split,
        "metadata": {},
    }


if __name__ == "__main__":
    unittest.main()
