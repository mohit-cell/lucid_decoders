from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lucid_decoders.tools.colab_recovery import collect_status, format_status


class ColabRecoveryTests(unittest.TestCase):
    def test_collect_status_summarizes_completed_chunks_and_resume_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            normalized_input = repo_root / "data/processed/en_de_full/all_trainable.jsonl"
            normalized_input.parent.mkdir(parents=True)
            normalized_input.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

            processed_dir = repo_root / "data/processed/en_de_full_features"
            chunks_dir = processed_dir / "chunks"
            chunks_dir.mkdir(parents=True)
            write_json(
                chunks_dir / "chunk_00000.report.json",
                {
                    "status": "completed",
                    "processed_examples": 2,
                    "skipped_examples": 0,
                    "token_rows": 10,
                    "sentence_rows": 2,
                    "sentence_head_rows": 384,
                },
            )
            (processed_dir / "token_features.parquet").write_bytes(b"token")

            status = collect_status(
                repo_root=repo_root,
                processed_dir=processed_dir,
                normalized_input=normalized_input,
                chunk_size=250,
                model_name="facebook/mbart-large-50-many-to-many-mmt",
                source_lang="en_XX",
                target_lang="de_DE",
                device="cuda",
            )

            self.assertEqual(status["normalized_input"]["line_count"], 2)
            self.assertEqual(status["chunks"]["completed_chunks"], 1)
            self.assertEqual(status["chunks"]["processed_examples"], 2)
            self.assertEqual(status["chunks"]["sentence_head_rows"], 384)
            self.assertTrue(status["merged_outputs"]["token"]["exists"])
            self.assertIn("--stage extract-chunked", status["resume_command"])

    def test_format_status_contains_resume_command(self) -> None:
        status = {
            "repo_root": "/content/drive/MyDrive/NLP/NLP_Project/lucid_decoders",
            "repo_exists": True,
            "drive_backed": True,
            "normalized_input": {
                "path": "/content/drive/MyDrive/NLP/NLP_Project/lucid_decoders/data/processed/en_de_full/all_trainable.jsonl",
                "exists": True,
                "line_count": 10,
            },
            "processed_dir": {
                "path": "/content/drive/MyDrive/NLP/NLP_Project/lucid_decoders/data/processed/en_de_full_features",
                "exists": True,
            },
            "chunks": {
                "exists": True,
                "report_files": 2,
                "completed_chunks": 2,
                "processed_examples": 500,
                "skipped_examples": 0,
                "token_rows": 1000,
                "sentence_rows": 500,
                "sentence_head_rows": 96000,
            },
            "merged_outputs": {
                "token": {"exists": True, "size": "1.0 MB"},
                "sentence": {"exists": True, "size": "10.0 KB"},
                "sentence_head": {"exists": True, "size": "5.0 MB"},
                "report": {"exists": True, "size": "1.0 KB"},
            },
            "final_report": {"processed_examples": 500, "completed_chunks": 2},
            "resume_command": "!PYTHONPATH=src python -m lucid_decoders.pipeline",
        }

        output = format_status(status)

        self.assertIn("Colab recovery status", output)
        self.assertIn("completed_chunks: 2", output)
        self.assertIn("Resume command", output)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

