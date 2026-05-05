"""Generate Kaggle commands for the 15k mBART extraction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lucid_decoders.tools.colab_recovery import count_lines, file_status, human_size, read_json_if_exists


DEFAULT_REPO_URL = "https://github.com/mohit-cell/lucid_decoders.git"
DEFAULT_BRANCH = "Mohit_dev"
DEFAULT_REPO_ROOT = "/kaggle/working/lucid_decoders"
DEFAULT_OUTPUT_DIR = "/kaggle/working/lucid_decoders_kaggle_outputs"
DEFAULT_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEFAULT_SOURCE_LANG = "en_XX"
DEFAULT_TARGET_LANG = "de_DE"
DEFAULT_TRAIN_PER_LABEL = 6537
DEFAULT_VALIDATION_PER_LABEL = 758
DEFAULT_TEST_PER_LABEL = 205
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHECKPOINT_EVERY = 5000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print and inspect the Kaggle 15k mBART extraction/training runbook."
    )
    parser.add_argument(
        "--mode",
        choices=["commands", "recovery", "status", "json"],
        default="commands",
        help="commands prints notebook cells, recovery prints restore steps, status inspects outputs, json emits status JSON.",
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-per-label", type=int, default=DEFAULT_TRAIN_PER_LABEL)
    parser.add_argument("--validation-per-label", type=int, default=DEFAULT_VALIDATION_PER_LABEL)
    parser.add_argument("--test-per-label", type=int, default=DEFAULT_TEST_PER_LABEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--head-train-jobs", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = KaggleRunConfig.from_args(args)
    if args.mode == "commands":
        print(format_commands(config))
    elif args.mode == "recovery":
        print(format_recovery(config))
    elif args.mode == "status":
        print(format_status(collect_status(config)))
    elif args.mode == "json":
        print(json.dumps(collect_status(config), indent=2))
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


class KaggleRunConfig:
    def __init__(
        self,
        *,
        repo_url: str,
        branch: str,
        repo_root: Path,
        output_dir: Path,
        model_name: str,
        source_lang: str,
        target_lang: str,
        device: str,
        train_per_label: int,
        validation_per_label: int,
        test_per_label: int,
        chunk_size: int,
        checkpoint_every: int,
        head_train_jobs: int,
    ) -> None:
        self.repo_url = repo_url
        self.branch = branch
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.device = device
        self.train_per_label = train_per_label
        self.validation_per_label = validation_per_label
        self.test_per_label = test_per_label
        self.chunk_size = chunk_size
        self.checkpoint_every = checkpoint_every
        self.head_train_jobs = head_train_jobs

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "KaggleRunConfig":
        return cls(
            repo_url=args.repo_url,
            branch=args.branch,
            repo_root=Path(args.repo_root),
            output_dir=Path(args.output_dir),
            model_name=args.model_name,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            device=args.device,
            train_per_label=args.train_per_label,
            validation_per_label=args.validation_per_label,
            test_per_label=args.test_per_label,
            chunk_size=args.chunk_size,
            checkpoint_every=args.checkpoint_every,
            head_train_jobs=args.head_train_jobs,
        )

    @property
    def total_examples(self) -> int:
        return 2 * (self.train_per_label + self.validation_per_label + self.test_per_label)

    @property
    def full_processed_dir(self) -> Path:
        return self.repo_root / "data/processed/en_de_full"

    @property
    def subset_dir(self) -> Path:
        return self.repo_root / "data/processed/en_de_subsets"

    @property
    def subset_path(self) -> Path:
        return self.subset_dir / f"balanced_{self.total_examples // 1000}k_sentence.jsonl"

    @property
    def subset_summary_path(self) -> Path:
        return self.subset_dir / f"balanced_{self.total_examples // 1000}k_sentence_summary.json"

    @property
    def feature_dir(self) -> Path:
        return self.repo_root / f"data/processed/en_de_{self.total_examples // 1000}k_features"

    @property
    def smoke_dir(self) -> Path:
        return self.repo_root / f"data/processed/en_de_{self.total_examples // 1000}k_smoke"

    @property
    def artifact_root(self) -> Path:
        return self.repo_root / f"artifacts/en_de_{self.total_examples // 1000}k"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def checkpoint_targets(self) -> list[int]:
        targets = list(range(self.checkpoint_every, self.total_examples, self.checkpoint_every))
        if self.total_examples not in targets:
            targets.append(self.total_examples)
        return targets


def format_commands(config: KaggleRunConfig) -> str:
    return "\n\n".join(
        [
            "# Cell 1: verify Kaggle GPU\n"
            "import torch\n"
            "print(torch.cuda.is_available())\n"
            "print(torch.cuda.device_count())\n"
            'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")',
            "# Cell 2: clone repo\n"
            "%cd /kaggle/working\n"
            f"!git clone {config.repo_url}\n"
            f"%cd {config.repo_root}\n"
            f"!git checkout {config.branch}\n"
            "!git pull",
            "# Cell 3: install dependencies\n"
            "!pip install -e .",
            "# Cell 4: verify raw WMT data\n"
            "!ls data/raw/wmt22\n"
            "!ls data/raw/wmt23\n"
            "!git submodule update --init --recursive",
            "# Cell 5: prepare normalized data\n"
            "!PYTHONPATH=src python -m lucid_decoders.pipeline \\\n"
            "  --stage prepare \\\n"
            "  --wmt22-root data/raw/wmt22 \\\n"
            "  --wmt23-root data/raw/wmt23 \\\n"
            "  --processed-dir data/processed/en_de_full\n"
            "!wc -l data/processed/en_de_full/*.jsonl\n"
            "!ls -lh data/processed/en_de_full",
            "# Cell 6: create balanced subset\n"
            "!mkdir -p data/processed/en_de_subsets\n"
            "!PYTHONPATH=src python -m lucid_decoders.data.sample_ende \\\n"
            "  --input data/processed/en_de_full/all_trainable.jsonl \\\n"
            f"  --output {relative_to_repo(config.subset_path, config.repo_root)} \\\n"
            f"  --train-per-label {config.train_per_label} \\\n"
            f"  --validation-per-label {config.validation_per_label} \\\n"
            f"  --test-per-label {config.test_per_label} \\\n"
            "  --seed 13 \\\n"
            "  --allow-positive-without-token-supervision \\\n"
            f"  --summary-output {relative_to_repo(config.subset_summary_path, config.repo_root)}\n"
            f"!wc -l {relative_to_repo(config.subset_path, config.repo_root)}\n"
            f"!cat {relative_to_repo(config.subset_summary_path, config.repo_root)}",
            "# Cell 7: smoke extraction\n"
            f"{extract_command(config, config.smoke_dir, max_examples=20, chunk_size=10)}\n"
            f"!ls -lh {relative_to_repo(config.smoke_dir, config.repo_root)}\n"
            f"!cat {relative_to_repo(config.smoke_dir, config.repo_root)}/mbart_extraction_report.json",
            *staged_extraction_cells(config),
            "# Cell 11: train logistic regression models\n"
            f"{train_command(config, 'train-sentence', 'logistic_regression')}\n"
            f"{train_command(config, 'train-heads', 'logistic_regression')}\n"
            f"{train_command(config, 'train-token', 'logistic_regression')}",
            "# Cell 12: optional sentence/token comparison models\n"
            "!for MODEL in random_forest mlp; do \\\n"
            "  PYTHONPATH=src python -m lucid_decoders.pipeline \\\n"
            "    --stage train-sentence \\\n"
            f"    --processed-dir {relative_to_repo(config.feature_dir, config.repo_root)} \\\n"
            f"    --artifacts-dir {relative_to_repo(config.artifact_root, config.repo_root)}/$MODEL \\\n"
            "    --model-type $MODEL \\\n"
            "    --seed 13; \\\n"
            "  PYTHONPATH=src python -m lucid_decoders.pipeline \\\n"
            "    --stage train-token \\\n"
            f"    --processed-dir {relative_to_repo(config.feature_dir, config.repo_root)} \\\n"
            f"    --artifacts-dir {relative_to_repo(config.artifact_root, config.repo_root)}/$MODEL \\\n"
            "    --model-type $MODEL \\\n"
            "    --seed 13; \\\n"
            "done",
            "# Cell 13: package Kaggle outputs before Save Version\n"
            f"!mkdir -p {config.output_dir}\n"
            f"!cp -r {relative_to_repo(config.subset_dir, config.repo_root)} {config.output_dir}/\n"
            f"!cp -r {relative_to_repo(config.feature_dir, config.repo_root)} {config.output_dir}/\n"
            f"!cp -r {relative_to_repo(config.artifact_root, config.repo_root)} {config.output_dir}/\n"
            f"!du -sh {config.output_dir}",
        ]
    )


def staged_extraction_cells(config: KaggleRunConfig) -> list[str]:
    cells: list[str] = []
    for index, target in enumerate(config.checkpoint_targets, start=8):
        label = f"{target // 1000}k"
        cells.append(
            f"# Cell {index}: extract through {label} and save a checkpoint\n"
            f"{extract_command(config, config.feature_dir, max_examples=target, chunk_size=config.chunk_size)}\n"
            f"!find {relative_to_repo(config.feature_dir, config.repo_root)}/chunks -name \"*.report.json\" | wc -l\n"
            f"!ls -lh {relative_to_repo(config.feature_dir, config.repo_root)}\n"
            f"{checkpoint_command(config, label)}"
        )
    return cells


def extract_command(
    config: KaggleRunConfig,
    processed_dir: Path,
    *,
    max_examples: int | None,
    chunk_size: int,
) -> str:
    lines = [
        "!PYTHONPATH=src python -m lucid_decoders.pipeline \\",
        "  --stage extract-chunked \\",
        f"  --normalized-input {relative_to_repo(config.subset_path, config.repo_root)} \\",
        f"  --processed-dir {relative_to_repo(processed_dir, config.repo_root)} \\",
        f"  --model-name {config.model_name} \\",
        f"  --source-lang {config.source_lang} \\",
        f"  --target-lang {config.target_lang} \\",
        f"  --device {config.device} \\",
        f"  --chunk-size {chunk_size}",
    ]
    if max_examples is not None:
        lines[-1] += " \\"
        lines.append(f"  --max-examples {max_examples}")
    return "\n".join(lines)


def train_command(config: KaggleRunConfig, stage: str, model_type: str) -> str:
    lines = [
        "!PYTHONPATH=src python -m lucid_decoders.pipeline \\",
        f"  --stage {stage} \\",
        f"  --processed-dir {relative_to_repo(config.feature_dir, config.repo_root)} \\",
        f"  --artifacts-dir {relative_to_repo(config.artifact_root, config.repo_root)}/{model_type} \\",
        f"  --model-type {model_type} \\",
        "  --seed 13",
    ]
    if stage == "train-heads":
        lines[-1] += " \\"
        lines.append(f"  --head-train-jobs {config.head_train_jobs}")
    return "\n".join(lines)


def checkpoint_command(config: KaggleRunConfig, label: str) -> str:
    archive = config.checkpoints_dir / f"en_de_{config.total_examples // 1000}k_chunks_checkpoint_{label}.tar.gz"
    return "\n".join(
        [
            f"!mkdir -p {config.checkpoints_dir}",
            f"!tar -czf {archive} \\",
            f"  {relative_to_repo(config.feature_dir, config.repo_root)}/chunks \\",
            f"  {relative_to_repo(config.subset_path, config.repo_root)} \\",
            f"  {relative_to_repo(config.subset_summary_path, config.repo_root)}",
            f"!ls -lh {archive}",
            f"!du -sh {config.checkpoints_dir}",
        ]
    )


def format_recovery(config: KaggleRunConfig) -> str:
    latest_archive = (
        f"/kaggle/input/YOUR_CHECKPOINT_DATASET/"
        f"en_de_{config.total_examples // 1000}k_chunks_checkpoint_10k.tar.gz"
    )
    return "\n\n".join(
        [
            "# Recovery Step 1: clone repo again after Kaggle reset\n"
            "%cd /kaggle/working\n"
            f"!git clone {config.repo_url}\n"
            f"%cd {config.repo_root}\n"
            f"!git checkout {config.branch}\n"
            "!pip install -e .",
            "# Recovery Step 2: add your checkpoint dataset as Kaggle input, then inspect it\n"
            "!find /kaggle/input -name \"en_de_*_chunks_checkpoint_*.tar.gz\" -print",
            "# Recovery Step 3: restore the latest checkpoint tarball into the repo root\n"
            "# Replace 10k with 5k or 15k if that is the checkpoint archive you saved.\n"
            f"!tar -xzf {latest_archive} -C {config.repo_root}",
            "# Recovery Step 4: verify restored chunks\n"
            f"!find {relative_to_repo(config.feature_dir, config.repo_root)}/chunks -name \"*.report.json\" | wc -l\n"
            "!lucid-kaggle-15k --mode status",
            "# Recovery Step 5: rerun the staged extraction cell for the next target, or rerun final extraction\n"
            f"{extract_command(config, config.feature_dir, max_examples=config.total_examples, chunk_size=config.chunk_size)}",
        ]
    )


def collect_status(config: KaggleRunConfig) -> dict[str, Any]:
    feature_report = read_json_if_exists(config.feature_dir / "mbart_extraction_report.json")
    chunk_reports = (
        len(list((config.feature_dir / "chunks").glob("*.report.json")))
        if (config.feature_dir / "chunks").exists()
        else 0
    )
    processed_from_chunks = chunk_reports * config.chunk_size
    return {
        "repo_root": path_status(config.repo_root),
        "kaggle_working": str(config.repo_root).startswith("/kaggle/working/"),
        "target_examples": config.total_examples,
        "chunk_size": config.chunk_size,
        "total_chunks": expected_chunks(config.total_examples, config.chunk_size),
        "checkpoint_targets": config.checkpoint_targets,
        "raw_data": {
            "wmt22": path_status(config.repo_root / "data/raw/wmt22"),
            "wmt23": path_status(config.repo_root / "data/raw/wmt23"),
        },
        "normalized_input": jsonl_status(config.full_processed_dir / "all_trainable.jsonl"),
        "subset": jsonl_status(config.subset_path),
        "subset_summary": read_json_if_exists(config.subset_summary_path),
        "features": {
            "dir": path_status(config.feature_dir),
            "token": file_status(config.feature_dir / "token_features.parquet"),
            "sentence": file_status(config.feature_dir / "sentence_features.parquet"),
            "sentence_head": file_status(config.feature_dir / "sentence_head_features.parquet"),
            "report": feature_report,
            "chunk_reports": chunk_reports,
            "processed_examples_estimate_from_chunks": min(processed_from_chunks, config.total_examples),
        },
        "artifacts": {
            "logistic_regression": path_status(config.artifact_root / "logistic_regression"),
            "random_forest": path_status(config.artifact_root / "random_forest"),
            "mlp": path_status(config.artifact_root / "mlp"),
        },
        "output_dir": path_status(config.output_dir),
        "checkpoints_dir": path_status(config.checkpoints_dir),
    }


def format_status(status: dict[str, Any]) -> str:
    features = status["features"]
    lines = [
        "Kaggle 15k run status",
        f"- kaggle_working: {status['kaggle_working']}",
        f"- target_examples: {status['target_examples']}",
        f"- total_chunks: {status['total_chunks']}",
        f"- checkpoint_targets: {status['checkpoint_targets']}",
        f"- repo_root_exists: {status['repo_root']['exists']}",
        f"- wmt22_exists: {status['raw_data']['wmt22']['exists']}",
        f"- wmt23_exists: {status['raw_data']['wmt23']['exists']}",
        f"- all_trainable_lines: {status['normalized_input']['line_count']}",
        f"- subset_lines: {status['subset']['line_count']}",
        f"- feature_chunk_reports: {features['chunk_reports']}",
        f"- processed_examples_estimate_from_chunks: {features['processed_examples_estimate_from_chunks']}",
        f"- token_features: exists={features['token']['exists']} size={features['token']['size']}",
        f"- sentence_features: exists={features['sentence']['exists']} size={features['sentence']['size']}",
        f"- sentence_head_features: exists={features['sentence_head']['exists']} size={features['sentence_head']['size']}",
        f"- extraction_processed_examples: {(features['report'] or {}).get('processed_examples')}",
        f"- output_dir_exists: {status['output_dir']['exists']}",
        f"- checkpoints_dir_exists: {status['checkpoints_dir']['exists']}",
    ]
    return "\n".join(lines)


def path_status(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def jsonl_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "line_count": count_lines(path) if path.exists() else None,
        "size": human_size(path.stat().st_size) if path.exists() else None,
    }


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def expected_chunks(total_examples: int, chunk_size: int) -> int:
    return (total_examples + chunk_size - 1) // chunk_size


if __name__ == "__main__":
    main()
