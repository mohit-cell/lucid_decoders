from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lucid_decoders.config import MBartConfig
from lucid_decoders.data.prepare_ende import prepare_ende_datasets
from lucid_decoders.data.validate_ende import raise_for_missing_data, validate_wmt_roots


STAGE_ORDER = ("prepare", "extract", "train-token", "train-sentence", "train-heads")
STAGE_CHOICES = (*STAGE_ORDER, "extract-chunked", "all")


def build_arg_parser() -> argparse.ArgumentParser:
    mbart_defaults = MBartConfig()
    parser = argparse.ArgumentParser(description="Run the WMT22/WMT23 en-de mBART attention pipeline.")
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGE_CHOICES,
        help="Pipeline stage to run. Use `all` for the full pipeline.",
    )
    parser.add_argument("--wmt22-root", default="data/raw/wmt22")
    parser.add_argument("--wmt23-root", default="data/raw/wmt23")
    parser.add_argument("--processed-dir", default="data/processed/en_de")
    parser.add_argument("--artifacts-dir", default="artifacts/en_de")
    parser.add_argument("--normalized-input", help="Override the normalized JSONL used by extraction.")
    parser.add_argument("--mqm-threshold", type=float, default=0.0)
    parser.add_argument("--skip-data-validation", action="store_true")
    parser.add_argument("--model-name", default=mbart_defaults.model_name)
    parser.add_argument("--source-lang", default=mbart_defaults.source_lang)
    parser.add_argument("--target-lang", default=mbart_defaults.target_lang)
    parser.add_argument("--max-source-length", type=int, default=mbart_defaults.max_source_length)
    parser.add_argument("--max-target-length", type=int, default=mbart_defaults.max_target_length)
    parser.add_argument("--device")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-examples", type=int, help="Smoke-run limit for the extraction stage.")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument(
        "--model-type",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "mlp"],
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-train-examples", type=int, default=20)
    parser.add_argument("--head-train-jobs", type=int, default=1)
    parser.add_argument(
        "--persist-head-models",
        choices=["best", "all", "none"],
        default="best",
        help="Persist only the best head model, every per-head model, or no head models.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for stage in resolve_stages(args.stage):
        if stage == "prepare":
            run_prepare(args, processed_dir)
        elif stage == "extract":
            run_extract(args, processed_dir)
        elif stage == "extract-chunked":
            run_extract_chunked(args, processed_dir)
        elif stage == "train-token":
            run_train_token(args, processed_dir, artifacts_dir)
        elif stage == "train-sentence":
            run_train_sentence(args, processed_dir, artifacts_dir)
        elif stage == "train-heads":
            run_train_heads(args, processed_dir, artifacts_dir)
        else:
            raise ValueError(f"Unsupported stage: {stage}")


def resolve_stages(stage: str) -> tuple[str, ...]:
    if stage == "all":
        return STAGE_ORDER
    return (stage,)


def run_prepare(args: argparse.Namespace, processed_dir: Path) -> None:
    if not args.skip_data_validation:
        raise_for_missing_data(validate_wmt_roots(args.wmt22_root, args.wmt23_root))
    summary = prepare_ende_datasets(
        wmt22_root=args.wmt22_root,
        wmt23_root=args.wmt23_root,
        output_dir=processed_dir,
        mqm_threshold=args.mqm_threshold,
    )
    print_stage("prepare", f"wrote {summary['all_trainable_examples']} trainable examples")


def run_extract(args: argparse.Namespace, processed_dir: Path) -> None:
    normalized_input = Path(args.normalized_input) if args.normalized_input else processed_dir / "all_trainable.jsonl"
    command = [
        "-m",
        "lucid_decoders.models.mbart_attention",
        "--input",
        str(normalized_input),
        "--token-output",
        str(processed_dir / "token_features.parquet"),
        "--sentence-output",
        str(processed_dir / "sentence_features.parquet"),
        "--sentence-head-output",
        str(processed_dir / "sentence_head_features.parquet"),
        "--model-name",
        args.model_name,
        "--source-lang",
        args.source_lang,
        "--target-lang",
        args.target_lang,
        "--max-source-length",
        str(args.max_source_length),
        "--max-target-length",
        str(args.max_target_length),
        "--topk",
        str(args.topk),
        "--report-output",
        str(processed_dir / "mbart_extraction_report.json"),
        "--require-sentence-label",
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.max_examples is not None:
        command.extend(["--max-examples", str(args.max_examples)])
    run_python(command)


def run_extract_chunked(args: argparse.Namespace, processed_dir: Path) -> None:
    normalized_input = Path(args.normalized_input) if args.normalized_input else processed_dir / "all_trainable.jsonl"
    command = [
        "-m",
        "lucid_decoders.models.mbart_attention_chunked",
        "--input",
        str(normalized_input),
        "--token-output",
        str(processed_dir / "token_features.parquet"),
        "--sentence-output",
        str(processed_dir / "sentence_features.parquet"),
        "--sentence-head-output",
        str(processed_dir / "sentence_head_features.parquet"),
        "--chunks-dir",
        str(processed_dir / "chunks"),
        "--chunk-size",
        str(args.chunk_size),
        "--resume",
        "--model-name",
        args.model_name,
        "--source-lang",
        args.source_lang,
        "--target-lang",
        args.target_lang,
        "--max-source-length",
        str(args.max_source_length),
        "--max-target-length",
        str(args.max_target_length),
        "--topk",
        str(args.topk),
        "--report-output",
        str(processed_dir / "mbart_extraction_report.json"),
        "--require-sentence-label",
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.max_examples is not None:
        command.extend(["--max-examples", str(args.max_examples)])
    run_python(command)


def run_train_token(args: argparse.Namespace, processed_dir: Path, artifacts_dir: Path) -> None:
    command = [
        "-m",
        "lucid_decoders.train.train_token_classifier",
        "--features",
        str(processed_dir / "token_features.parquet"),
        "--artifacts-dir",
        str(artifacts_dir / "token_classifier"),
        "--model-type",
        args.model_type,
        "--seed",
        str(args.seed),
    ]
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    run_python(command)


def run_train_sentence(args: argparse.Namespace, processed_dir: Path, artifacts_dir: Path) -> None:
    command = [
        "-m",
        "lucid_decoders.train.train_sentence_classifier",
        "--features",
        str(processed_dir / "sentence_features.parquet"),
        "--artifacts-dir",
        str(artifacts_dir / "sentence_classifier"),
        "--model-type",
        args.model_type,
        "--seed",
        str(args.seed),
    ]
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    run_python(command)


def run_train_heads(args: argparse.Namespace, processed_dir: Path, artifacts_dir: Path) -> None:
    command = [
        "-m",
        "lucid_decoders.train.train_sentence_head_classifier",
        "--features",
        str(processed_dir / "sentence_head_features.parquet"),
        "--artifacts-dir",
        str(artifacts_dir / "sentence_head_classifier"),
        "--model-type",
        args.model_type,
        "--seed",
        str(args.seed),
        "--min-train-examples",
        str(args.min_train_examples),
        "--resume",
        "--persist-head-models",
        args.persist_head_models,
    ]
    if args.head_train_jobs != 1:
        command.extend(["--n-jobs", str(args.head_train_jobs)])
    if args.threshold is not None:
        command.extend(["--threshold", str(args.threshold)])
    run_python(command)


def run_python(args: list[str]) -> None:
    printable = " ".join([sys.executable, *args])
    print_stage("run", printable)
    subprocess.run([sys.executable, *args], check=True)


def print_stage(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


if __name__ == "__main__":
    main()
