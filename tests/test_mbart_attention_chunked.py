from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from lucid_decoders.config import FeatureConfig, MBartConfig
from lucid_decoders.features.contracts import SENTENCE_HEAD_REQUIRED_COLUMNS, TOKEN_REQUIRED_COLUMNS
from lucid_decoders.models.mbart_attention_chunked import run_chunked_extraction
from lucid_decoders.schemas import TranslationExample


class ChunkedExtractionTests(unittest.TestCase):
    def test_chunked_extraction_merges_unique_examples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = run_fake_chunked(root, examples=build_examples(5), chunk_size=2)

            self.assertEqual(report["completed_chunks"], 3)
            self.assertEqual(report["processed_examples"], 5)
            token_frame = pd.read_parquet(root / "token.parquet")
            sentence_frame = pd.read_parquet(root / "sentence.parquet")
            sentence_head_frame = pd.read_parquet(root / "sentence_head.parquet")
            self.assertEqual(token_frame["example_id"].nunique(), 5)
            self.assertEqual(sentence_frame["example_id"].nunique(), 5)
            self.assertEqual(sentence_head_frame["example_id"].nunique(), 5)

    def test_resume_skips_completed_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            examples = build_examples(4)
            first_factory = CountingExtractorFactory()
            run_fake_chunked(root, examples=examples, chunk_size=2, extractor_factory=first_factory)
            self.assertEqual(first_factory.extract_calls, 4)

            second_factory = CountingExtractorFactory()
            report = run_fake_chunked(root, examples=examples, chunk_size=2, extractor_factory=second_factory)
            self.assertEqual(second_factory.extract_calls, 0)
            self.assertEqual(report["completed_chunks"], 2)


def run_fake_chunked(
    root: Path,
    examples: list[TranslationExample],
    chunk_size: int,
    extractor_factory=None,
) -> dict:
    return run_chunked_extraction(
        examples=examples,
        input_path="fake.jsonl",
        mbart_config=MBartConfig(device="cpu"),
        feature_config=FeatureConfig(),
        chunks_dir=root / "chunks",
        token_output=root / "token.parquet",
        sentence_output=root / "sentence.parquet",
        sentence_head_output=root / "sentence_head.parquet",
        report_output=root / "report.json",
        chunk_size=chunk_size,
        resume=True,
        require_sentence_label=True,
        extractor_factory=extractor_factory or CountingExtractorFactory(),
        token_row_builder=fake_token_rows,
        sentence_head_row_builder=fake_sentence_head_rows,
    )


class CountingExtractorFactory:
    def __init__(self) -> None:
        self.extract_calls = 0

    def __call__(self, config: MBartConfig) -> "CountingExtractorFactory":
        return self

    def extract(self, example: TranslationExample) -> SimpleNamespace:
        self.extract_calls += 1
        return SimpleNamespace(
            example_id=example.example_id,
            language_pair=example.language_pair,
            split=example.split,
            source_text=example.source_text,
            hypothesis_text=example.hypothesis_text,
            sentence_label=example.sentence_label,
        )


def fake_token_rows(extraction: SimpleNamespace, feature_config: FeatureConfig) -> list[dict]:
    rows = []
    for token_index in range(2):
        row = {}
        for column in TOKEN_REQUIRED_COLUMNS:
            row[column] = numeric_value(column, extraction.sentence_label or 0, token_index)
        row.update(
            {
                "example_id": extraction.example_id,
                "language_pair": extraction.language_pair,
                "split": extraction.split,
                "source_text": extraction.source_text,
                "hypothesis_text": extraction.hypothesis_text,
                "sentence_label": extraction.sentence_label,
                "token_label": int(extraction.sentence_label or 0),
                "token_index": token_index,
                "token_text": f"tok{token_index}",
                "token_start_char": token_index,
                "token_end_char": token_index + 1,
            }
        )
        rows.append(row)
    return rows


def fake_sentence_head_rows(extraction: SimpleNamespace, feature_config: FeatureConfig) -> list[dict]:
    row = {}
    for column in SENTENCE_HEAD_REQUIRED_COLUMNS:
        row[column] = numeric_value(column, extraction.sentence_label or 0, 0)
    row.update(
        {
            "example_id": extraction.example_id,
            "language_pair": extraction.language_pair,
            "split": extraction.split,
            "source_text": extraction.source_text,
            "hypothesis_text": extraction.hypothesis_text,
            "sentence_label": extraction.sentence_label,
            "layer_id": 0,
            "head_id": 0,
        }
    )
    return [row]


def numeric_value(column: str, label: int, offset: int) -> float:
    if column in {"layer_id", "head_id", "token_index", "token_start_char", "token_end_char"}:
        return offset
    if column in {"source_length_tokens", "target_length_tokens"}:
        return 2.0
    return float(label) + 0.01 * ((len(column) + offset) % 7)


def build_examples(count: int) -> list[TranslationExample]:
    return [
        TranslationExample(
            example_id=f"ex-{idx}",
            source_text=f"source {idx}",
            hypothesis_text=f"target {idx}",
            sentence_label=idx % 2,
            language_pair="en-de",
            split="train" if idx < count - 1 else "test",
        )
        for idx in range(count)
    ]


if __name__ == "__main__":
    unittest.main()
