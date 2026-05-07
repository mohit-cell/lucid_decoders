from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from lucid_decoders.tools.local_run import (
    DEFAULT_MODEL_TYPES,
    head_classifier_complete,
    is_process_running,
    read_json_if_exists,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a resumable local run and print the resume command.")
    parser.add_argument("--run-id", default="en_de_50k")
    parser.add_argument("--processed-dir", default="data/processed/en_de_50k")
    parser.add_argument("--artifacts-dir", default="artifacts/en_de_50k")
    parser.add_argument("--model-types", nargs="+", default=list(DEFAULT_MODEL_TYPES))
    parser.add_argument("--persist-head-models", choices=["best", "all", "none"], default="best")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    status = collect_status(args)
    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(format_status(status))


def collect_status(args: argparse.Namespace) -> dict[str, Any]:
    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)
    logs_dir = artifacts_dir / "logs"
    state = read_json_if_exists(artifacts_dir / "run_state.json")
    lock = read_json_if_exists(logs_dir / "run.lock")
    lock_pid = lock.get("pid") if isinstance(lock, dict) else None
    chunks = chunk_status(processed_dir / "chunks")
    return {
        "run_id": args.run_id,
        "processed_dir": str(processed_dir),
        "artifacts_dir": str(artifacts_dir),
        "run_state": state,
        "lock": {
            "path": str(logs_dir / "run.lock"),
            "exists": (logs_dir / "run.lock").exists(),
            "pid": lock_pid,
            "active": bool(isinstance(lock_pid, int) and is_process_running(lock_pid)),
        },
        "chunks": chunks,
        "outputs": output_status(processed_dir, artifacts_dir, args.model_types, args.persist_head_models),
        "resume_command": build_resume_command(args, state),
    }


def chunk_status(chunks_dir: Path) -> dict[str, Any]:
    reports = []
    if chunks_dir.exists():
        for path in sorted(chunks_dir.glob("*.report.json")):
            payload = read_json_if_exists(path)
            if payload:
                reports.append(payload)
    completed = [report for report in reports if report.get("status") == "completed"]
    return {
        "chunks_dir": str(chunks_dir),
        "exists": chunks_dir.exists(),
        "report_files": len(reports),
        "completed_chunks": len(completed),
        "processed_examples": sum_int(completed, "processed_examples"),
        "skipped_examples": sum_int(completed, "skipped_examples"),
        "token_rows": sum_int(completed, "token_rows"),
        "sentence_rows": sum_int(completed, "sentence_rows"),
        "sentence_head_rows": sum_int(completed, "sentence_head_rows"),
    }


def output_status(
    processed_dir: Path,
    artifacts_dir: Path,
    model_types: list[str],
    persist_head_models: str,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {
        "sample_summary": file_status(processed_dir / "sample_summary.json"),
        "token_features": file_status(processed_dir / "token_features.parquet"),
        "sentence_features": file_status(processed_dir / "sentence_features.parquet"),
        "sentence_head_features": file_status(processed_dir / "sentence_head_features.parquet"),
        "extraction_report": file_status(processed_dir / "mbart_extraction_report.json"),
        "run_summary": file_status(artifacts_dir / "run_summary.json"),
        "models": {},
    }
    for model_type in model_types:
        model_dir = artifacts_dir / model_type
        outputs["models"][model_type] = {
            "token": classifier_status(model_dir / "token_classifier"),
            "sentence": classifier_status(model_dir / "sentence_classifier"),
            "heads": head_classifier_complete(
                model_dir / "sentence_head_classifier",
                require_best_model=persist_head_models in {"best", "all"},
            ),
        }
    return outputs


def classifier_status(path: Path) -> bool:
    return all(
        item.exists() and item.stat().st_size > 0
        for item in [path / "metrics.json", path / "model.pkl", path / "test_predictions.parquet"]
    )


def file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def sum_int(items: list[dict[str, Any]], key: str) -> int:
    return sum(int(item.get(key, 0)) for item in items if isinstance(item.get(key, 0), int))


def build_resume_command(args: argparse.Namespace, state: dict[str, Any] | None = None) -> str:
    python = Path(sys.executable)
    config = state.get("config", {}) if isinstance(state, dict) else {}
    counts = config.get("counts", {}) if isinstance(config.get("counts", {}), dict) else {}
    model_types = config.get("model_types", args.model_types)
    if isinstance(model_types, str):
        model_types = [model_types]
    return " ".join(
        [
            str(python),
            "-m",
            "lucid_decoders.tools.local_run",
            "--run-id",
            args.run_id,
            "--processed-dir",
            str(config.get("processed_dir", args.processed_dir)),
            "--artifacts-dir",
            str(config.get("artifacts_dir", args.artifacts_dir)),
            "--normalized-source",
            str(config.get("normalized_source", "data/processed/en_de/all_trainable.jsonl")),
            "--device",
            str(config.get("device", "cuda")),
            "--chunk-size",
            str(config.get("chunk_size", 250)),
            "--seed",
            str(config.get("seed", 13)),
            "--train-per-label",
            str(counts.get("train_per_label", 24037)),
            "--validation-per-label",
            str(counts.get("validation_per_label", 758)),
            "--test-per-label",
            str(counts.get("test_per_label", 205)),
            "--head-train-jobs",
            str(config.get("head_train_jobs", 8)),
            "--persist-head-models",
            str(config.get("persist_head_models", args.persist_head_models)),
            "--model-types",
            *[str(model_type) for model_type in model_types],
        ]
    )


def format_status(status: dict[str, Any]) -> str:
    chunks = status["chunks"]
    lock = status["lock"]
    lines = [
        "Local run status",
        f"- run_id: {status['run_id']}",
        f"- processed_dir: {status['processed_dir']}",
        f"- artifacts_dir: {status['artifacts_dir']}",
        f"- active_lock: {lock['active']} pid={lock['pid']}",
        f"- completed_chunks: {chunks['completed_chunks']} from {chunks['report_files']} reports",
        f"- processed_examples_from_chunks: {chunks['processed_examples']}",
        f"- token_rows_from_chunks: {chunks['token_rows']}",
        f"- sentence_rows_from_chunks: {chunks['sentence_rows']}",
        f"- sentence_head_rows_from_chunks: {chunks['sentence_head_rows']}",
        "",
        "Resume command",
        status["resume_command"],
    ]
    run_state = status.get("run_state") or {}
    stages = run_state.get("stages", {})
    if stages:
        lines.extend(["", "Stages"])
        for name, info in stages.items():
            lines.append(f"- {name}: {info.get('status')} - {info.get('message')}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
