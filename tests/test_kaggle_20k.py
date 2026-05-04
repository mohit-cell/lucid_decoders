from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lucid_decoders.tools.kaggle_20k import (
    KaggleRunConfig,
    collect_status,
    format_commands,
    train_command,
)


class Kaggle20kTests(unittest.TestCase):
    def test_format_commands_contains_expected_20k_pipeline_steps(self) -> None:
        config = build_config(Path("/kaggle/working/lucid_decoders"))

        output = format_commands(config)

        self.assertIn("balanced_20k_sentence.jsonl", output)
        self.assertIn("--train-per-label 9037", output)
        self.assertIn("--validation-per-label 758", output)
        self.assertIn("--test-per-label 205", output)
        self.assertIn("--stage extract-chunked", output)
        self.assertIn("--stage train-heads", output)
        self.assertIn("/kaggle/working/lucid_decoders_kaggle_outputs", output)

    def test_train_head_command_adds_parallel_jobs(self) -> None:
        config = build_config(Path("/kaggle/working/lucid_decoders"))

        command = train_command(config, "train-heads", "logistic_regression")

        self.assertIn("--head-train-jobs 4", command)

    def test_collect_status_counts_subset_lines_and_chunk_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            subset_path = repo_root / "data/processed/en_de_subsets/balanced_20k_sentence.jsonl"
            subset_path.parent.mkdir(parents=True)
            subset_path.write_text("{}\n{}\n", encoding="utf-8")
            full_path = repo_root / "data/processed/en_de_full/all_trainable.jsonl"
            full_path.parent.mkdir(parents=True)
            full_path.write_text("{}\n", encoding="utf-8")
            feature_dir = repo_root / "data/processed/en_de_20k_features"
            chunks_dir = feature_dir / "chunks"
            chunks_dir.mkdir(parents=True)
            (chunks_dir / "chunk_00000.report.json").write_text("{}", encoding="utf-8")
            (feature_dir / "mbart_extraction_report.json").write_text(
                json.dumps({"processed_examples": 2}),
                encoding="utf-8",
            )
            config = build_config(repo_root)

            status = collect_status(config)

            self.assertEqual(status["normalized_input"]["line_count"], 1)
            self.assertEqual(status["subset"]["line_count"], 2)
            self.assertEqual(status["features"]["chunk_reports"], 1)
            self.assertEqual(status["features"]["report"]["processed_examples"], 2)


def build_config(repo_root: Path) -> KaggleRunConfig:
    return KaggleRunConfig(
        repo_url="https://github.com/mohit-cell/lucid_decoders.git",
        branch="Mohit_dev",
        repo_root=repo_root,
        output_dir=Path("/kaggle/working/lucid_decoders_kaggle_outputs"),
        model_name="facebook/mbart-large-50-many-to-many-mmt",
        source_lang="en_XX",
        target_lang="de_DE",
        device="cuda",
        train_per_label=9037,
        validation_per_label=758,
        test_per_label=205,
        chunk_size=250,
        head_train_jobs=4,
    )


if __name__ == "__main__":
    unittest.main()

