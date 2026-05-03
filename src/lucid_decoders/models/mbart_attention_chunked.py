from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lucid_decoders.config import FeatureConfig, MBartConfig
from lucid_decoders.features.contracts import validate_feature_frame
from lucid_decoders.features.sentence_head_features import build_sentence_head_feature_rows
from lucid_decoders.features.sentence_features import build_sentence_feature_frame
from lucid_decoders.features.token_features import build_token_feature_rows
from lucid_decoders.io import write_table
from lucid_decoders.models.mbart_attention import (
    MBartAttentionExtractor,
    load_examples,
    validate_example_for_extraction,
)
from lucid_decoders.schemas import AttentionExtraction, TranslationExample


ExtractorFactory = Callable[[MBartConfig], Any]
TokenRowBuilder = Callable[[AttentionExtraction, FeatureConfig], list[dict[str, Any]]]
SentenceHeadRowBuilder = Callable[[AttentionExtraction, FeatureConfig], list[dict[str, Any]]]


@dataclass(slots=True)
class ChunkPaths:
    token: Path
    sentence: Path
    sentence_head: Path | None
    report: Path


@dataclass(slots=True)
class ChunkReport:
    chunk_id: int
    start_index: int
    end_index: int
    total_examples: int
    processed_examples: int
    skipped_examples: int
    token_rows: int
    sentence_rows: int
    sentence_head_rows: int
    skipped: list[dict[str, str]]
    status: str = "completed"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract mBART attention features in resumable chunks.")
    parser.add_argument("--input", required=True, help="Normalized JSONL from the preprocessing step.")
    parser.add_argument("--token-output", required=True, help="Path to merged token-level feature table.")
    parser.add_argument("--sentence-output", required=True, help="Path to merged sentence-level feature table.")
    parser.add_argument("--sentence-head-output", help="Path to merged sentence-level layer/head feature table.")
    parser.add_argument("--chunks-dir", required=True, help="Directory for per-chunk feature files.")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--resume", action="store_true", help="Skip completed chunks whose outputs already exist.")
    parser.add_argument("--model-name", default="facebook/mbart-large-50-many-to-many-mmt")
    parser.add_argument("--source-lang", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--device")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument("--require-sentence-label", action="store_true")
    parser.add_argument("--require-token-labels", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    mbart_config = MBartConfig(
        model_name=args.model_name,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        device=args.device,
    )
    feature_config = FeatureConfig(topk=args.topk)
    examples = load_examples(args.input)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    run_chunked_extraction(
        examples=examples,
        input_path=str(args.input),
        mbart_config=mbart_config,
        feature_config=feature_config,
        chunks_dir=Path(args.chunks_dir),
        token_output=Path(args.token_output),
        sentence_output=Path(args.sentence_output),
        sentence_head_output=Path(args.sentence_head_output) if args.sentence_head_output else None,
        report_output=Path(args.report_output),
        chunk_size=args.chunk_size,
        resume=args.resume,
        fail_on_invalid=args.fail_on_invalid,
        require_sentence_label=args.require_sentence_label,
        require_token_labels=args.require_token_labels,
    )


def run_chunked_extraction(
    examples: Sequence[TranslationExample],
    input_path: str | None,
    mbart_config: MBartConfig,
    feature_config: FeatureConfig,
    chunks_dir: Path,
    token_output: Path,
    sentence_output: Path,
    sentence_head_output: Path | None,
    report_output: Path,
    chunk_size: int,
    resume: bool,
    fail_on_invalid: bool = False,
    require_sentence_label: bool = False,
    require_token_labels: bool = False,
    extractor_factory: ExtractorFactory = MBartAttentionExtractor,
    token_row_builder: TokenRowBuilder = build_token_feature_rows,
    sentence_head_row_builder: SentenceHeadRowBuilder = build_sentence_head_feature_rows,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    extractor = None
    chunk_reports: list[ChunkReport] = []
    for chunk_id, start_index in enumerate(range(0, len(examples), chunk_size)):
        end_index = min(start_index + chunk_size, len(examples))
        paths = build_chunk_paths(chunks_dir, chunk_id, include_sentence_head=sentence_head_output is not None)
        if resume and is_completed_chunk(paths):
            chunk_reports.append(read_chunk_report(paths.report))
            continue

        if extractor is None:
            extractor = extractor_factory(mbart_config)
        chunk_report = process_chunk(
            examples=list(examples[start_index:end_index]),
            extractor=extractor,
            feature_config=feature_config,
            paths=paths,
            chunk_id=chunk_id,
            start_index=start_index,
            end_index=end_index,
            fail_on_invalid=fail_on_invalid,
            require_sentence_label=require_sentence_label,
            require_token_labels=require_token_labels,
            token_row_builder=token_row_builder,
            sentence_head_row_builder=sentence_head_row_builder,
        )
        chunk_reports.append(chunk_report)

    if not chunk_reports and examples:
        raise ValueError("No chunks were processed or resumed.")

    merge_chunk_tables(
        reports=chunk_reports,
        chunks_dir=chunks_dir,
        token_output=token_output,
        sentence_output=sentence_output,
        sentence_head_output=sentence_head_output,
    )
    report = build_final_report(
        examples=examples,
        chunk_reports=chunk_reports,
        input_path=input_path,
        chunks_dir=chunks_dir,
        token_output=token_output,
        sentence_output=sentence_output,
        sentence_head_output=sentence_head_output,
        chunk_size=chunk_size,
    )
    atomic_write_json(report, report_output)
    return report


def process_chunk(
    examples: list[TranslationExample],
    extractor: Any,
    feature_config: FeatureConfig,
    paths: ChunkPaths,
    chunk_id: int,
    start_index: int,
    end_index: int,
    fail_on_invalid: bool,
    require_sentence_label: bool,
    require_token_labels: bool,
    token_row_builder: TokenRowBuilder,
    sentence_head_row_builder: SentenceHeadRowBuilder,
) -> ChunkReport:
    token_rows: list[dict[str, Any]] = []
    sentence_head_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, str]] = []

    for example in examples:
        invalid_reason = validate_example_for_extraction(
            example,
            require_sentence_label=require_sentence_label,
            require_token_labels=require_token_labels,
        )
        if invalid_reason:
            skipped_rows.append({"example_id": example.example_id, "reason": invalid_reason})
            if fail_on_invalid:
                raise ValueError(f"Invalid example {example.example_id}: {invalid_reason}")
            continue
        extraction = extractor.extract(example)
        token_rows.extend(token_row_builder(extraction, feature_config))
        if paths.sentence_head is not None:
            sentence_head_rows.extend(sentence_head_row_builder(extraction, feature_config))

    if not token_rows:
        raise ValueError(f"Chunk {chunk_id} produced no token rows.")

    token_frame = pd.DataFrame(token_rows)
    sentence_frame = build_sentence_feature_frame(token_frame)
    validate_feature_frame(token_frame, "token")
    validate_feature_frame(sentence_frame, "sentence")
    atomic_write_table(token_frame, paths.token)
    atomic_write_table(sentence_frame, paths.sentence)

    if paths.sentence_head is not None:
        sentence_head_frame = pd.DataFrame(sentence_head_rows)
        validate_feature_frame(sentence_head_frame, "sentence_head")
        atomic_write_table(sentence_head_frame, paths.sentence_head)
        sentence_head_count = len(sentence_head_frame)
    else:
        sentence_head_count = 0

    report = ChunkReport(
        chunk_id=chunk_id,
        start_index=start_index,
        end_index=end_index,
        total_examples=len(examples),
        processed_examples=len({row["example_id"] for row in token_rows}),
        skipped_examples=len(skipped_rows),
        token_rows=len(token_frame),
        sentence_rows=len(sentence_frame),
        sentence_head_rows=sentence_head_count,
        skipped=skipped_rows,
    )
    atomic_write_json(asdict(report), paths.report)
    return report


def merge_chunk_tables(
    reports: list[ChunkReport],
    chunks_dir: Path,
    token_output: Path,
    sentence_output: Path,
    sentence_head_output: Path | None,
) -> None:
    completed = sorted(reports, key=lambda report: report.chunk_id)
    token_frames = [pd.read_parquet(build_chunk_paths(chunks_dir, report.chunk_id, sentence_head_output is not None).token) for report in completed]
    sentence_frames = [pd.read_parquet(build_chunk_paths(chunks_dir, report.chunk_id, sentence_head_output is not None).sentence) for report in completed]
    token_frame = pd.concat(token_frames, ignore_index=True) if token_frames else pd.DataFrame()
    sentence_frame = pd.concat(sentence_frames, ignore_index=True) if sentence_frames else pd.DataFrame()
    validate_feature_frame(token_frame, "token")
    validate_feature_frame(sentence_frame, "sentence")
    atomic_write_table(token_frame, token_output)
    atomic_write_table(sentence_frame, sentence_output)

    if sentence_head_output is not None:
        sentence_head_frames = [
            pd.read_parquet(build_chunk_paths(chunks_dir, report.chunk_id, True).sentence_head)
            for report in completed
        ]
        sentence_head_frame = pd.concat(sentence_head_frames, ignore_index=True) if sentence_head_frames else pd.DataFrame()
        validate_feature_frame(sentence_head_frame, "sentence_head")
        atomic_write_table(sentence_head_frame, sentence_head_output)


def build_final_report(
    examples: Sequence[TranslationExample],
    chunk_reports: list[ChunkReport],
    input_path: str | None,
    chunks_dir: Path,
    token_output: Path,
    sentence_output: Path,
    sentence_head_output: Path | None,
    chunk_size: int,
) -> dict[str, Any]:
    completed = sorted(chunk_reports, key=lambda report: report.chunk_id)
    return {
        "input": input_path,
        "total_examples": len(examples),
        "processed_examples": sum(report.processed_examples for report in completed),
        "skipped_examples": sum(report.skipped_examples for report in completed),
        "skipped": [item for report in completed for item in report.skipped],
        "token_rows": sum(report.token_rows for report in completed),
        "sentence_rows": sum(report.sentence_rows for report in completed),
        "sentence_head_rows": sum(report.sentence_head_rows for report in completed),
        "chunk_size": chunk_size,
        "chunks_dir": str(chunks_dir),
        "completed_chunks": len(completed),
        "chunks": [asdict(report) for report in completed],
        "outputs": {
            "token": str(token_output),
            "sentence": str(sentence_output),
            "sentence_head": str(sentence_head_output) if sentence_head_output else None,
        },
    }


def build_chunk_paths(chunks_dir: Path, chunk_id: int, include_sentence_head: bool) -> ChunkPaths:
    stem = f"chunk_{chunk_id:05d}"
    return ChunkPaths(
        token=chunks_dir / f"{stem}.token.parquet",
        sentence=chunks_dir / f"{stem}.sentence.parquet",
        sentence_head=chunks_dir / f"{stem}.sentence_head.parquet" if include_sentence_head else None,
        report=chunks_dir / f"{stem}.report.json",
    )


def is_completed_chunk(paths: ChunkPaths) -> bool:
    required = [paths.token, paths.sentence, paths.report]
    if paths.sentence_head is not None:
        required.append(paths.sentence_head)
    if not all(path.exists() for path in required):
        return False
    report = read_chunk_report(paths.report)
    return report.status == "completed"


def read_chunk_report(path: Path) -> ChunkReport:
    return ChunkReport(**json.loads(path.read_text(encoding="utf-8")))


def atomic_write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    write_table(frame, temp_path)
    temp_path.replace(path)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(path)


if __name__ == "__main__":
    main()
