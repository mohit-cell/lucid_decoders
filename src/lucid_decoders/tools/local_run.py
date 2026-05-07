from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from lucid_decoders.io import write_json_atomic, write_text_atomic


DEFAULT_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEFAULT_SOURCE_LANG = "en_XX"
DEFAULT_TARGET_LANG = "de_DE"
DEFAULT_MODEL_TYPES = ("logistic_regression", "random_forest", "mlp")
BASE_STAGES = ("env-check", "sample", "extract", "report")
TRAIN_KINDS = ("train-token", "train-sentence", "train-heads")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a resumable local en-de mBART experiment.")
    parser.add_argument("--run-id", default="en_de_50k")
    parser.add_argument("--processed-dir", default="data/processed/en_de_50k")
    parser.add_argument("--artifacts-dir", default="artifacts/en_de_50k")
    parser.add_argument("--normalized-source", default="data/processed/en_de/all_trainable.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-per-label", type=int, default=24037)
    parser.add_argument("--validation-per-label", type=int, default=758)
    parser.add_argument("--test-per-label", type=int, default=205)
    parser.add_argument("--model-types", nargs="+", default=list(DEFAULT_MODEL_TYPES))
    parser.add_argument("--head-train-jobs", type=int, default=8)
    parser.add_argument("--min-train-examples", type=int, default=20)
    parser.add_argument(
        "--persist-head-models",
        choices=["best", "all", "none"],
        default="best",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[*BASE_STAGES, *TRAIN_KINDS, "all"],
        default=["all"],
        help="Stages to run. `all` expands to env-check, sample, extract, all training, report.",
    )
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_local(args)


def run_local(args: argparse.Namespace) -> dict[str, Any]:
    processed_dir = Path(args.processed_dir)
    artifacts_dir = Path(args.artifacts_dir)
    logs_dir = artifacts_dir / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    state_path = artifacts_dir / "run_state.json"
    lock_path = logs_dir / "run.lock"
    state = load_state(state_path) if args.resume else None
    if state is None:
        state = new_state(args)
    state.setdefault("stages", {})
    state.setdefault("events", [])

    acquire_lock(lock_path, state, logs_dir)
    try:
        append_event(state, "run_start", "Local run started or resumed.")
        save_state(state, state_path)
        append_run_log(logs_dir, "Run started or resumed.")
        ensure_change_log(logs_dir)

        for stage in expand_stages(args.stages, args.model_types):
            run_stage(stage, args, processed_dir, artifacts_dir, logs_dir, state, state_path)

        append_event(state, "run_complete", "Local run completed.")
        save_state(state, state_path)
        append_run_log(logs_dir, "Run completed.")
        return state
    finally:
        release_lock(lock_path)


def run_stage(
    stage: str,
    args: argparse.Namespace,
    processed_dir: Path,
    artifacts_dir: Path,
    logs_dir: Path,
    state: dict[str, Any],
    state_path: Path,
) -> None:
    if args.resume and stage_outputs_complete(stage, processed_dir, artifacts_dir, args):
        mark_stage(state, stage, "skipped", "Required outputs already exist.")
        save_state(state, state_path)
        write_stage_status(logs_dir, state)
        append_run_log(logs_dir, f"Skipped `{stage}` because outputs are complete.")
        return

    mark_stage(state, stage, "running", "Stage started.")
    save_state(state, state_path)
    write_stage_status(logs_dir, state)
    append_run_log(logs_dir, f"Started `{stage}`.")

    if stage == "env-check":
        run_environment_check(args, artifacts_dir)
    elif stage == "sample":
        run_logged_command(stage, sample_command(args, processed_dir), logs_dir, state, state_path, args.heartbeat_seconds)
    elif stage == "extract":
        run_logged_command(stage, extract_command(args, processed_dir, artifacts_dir), logs_dir, state, state_path, args.heartbeat_seconds)
    elif stage.startswith("train-token:"):
        model_type = stage.split(":", 1)[1]
        run_logged_command(stage, train_command(args, processed_dir, artifacts_dir, model_type, "train-token"), logs_dir, state, state_path, args.heartbeat_seconds)
    elif stage.startswith("train-sentence:"):
        model_type = stage.split(":", 1)[1]
        run_logged_command(stage, train_command(args, processed_dir, artifacts_dir, model_type, "train-sentence"), logs_dir, state, state_path, args.heartbeat_seconds)
    elif stage.startswith("train-heads:"):
        model_type = stage.split(":", 1)[1]
        run_logged_command(stage, train_command(args, processed_dir, artifacts_dir, model_type, "train-heads"), logs_dir, state, state_path, args.heartbeat_seconds)
    elif stage == "report":
        collect_run_report(processed_dir, artifacts_dir, args)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    mark_stage(state, stage, "completed", "Stage completed.")
    save_state(state, state_path)
    write_stage_status(logs_dir, state)
    append_run_log(logs_dir, f"Completed `{stage}`.")


def run_logged_command(
    stage: str,
    command: list[str],
    logs_dir: Path,
    state: dict[str, Any],
    state_path: Path,
    heartbeat_seconds: int,
) -> None:
    safe_stage = stage.replace(":", "_")
    stdout_path = logs_dir / f"{safe_stage}.stdout.log"
    stderr_path = logs_dir / f"{safe_stage}.stderr.log"
    state["stages"][stage]["command"] = command
    state["stages"][stage]["stdout"] = str(stdout_path)
    state["stages"][stage]["stderr"] = str(stderr_path)
    save_state(state, state_path)

    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n[{timestamp()}] command: {' '.join(command)}\n")
        stdout.flush()
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        state["stages"][stage]["pid"] = process.pid
        save_state(state, state_path)
        while True:
            return_code = process.poll()
            write_heartbeat(logs_dir, state, stage, pid=process.pid)
            if return_code is not None:
                state["stages"][stage]["return_code"] = return_code
                save_state(state, state_path)
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)
                return
            time.sleep(max(1, heartbeat_seconds))


def run_environment_check(args: argparse.Namespace, artifacts_dir: Path) -> None:
    payload: dict[str, Any] = {
        "python": sys.executable,
        "platform": platform.platform(),
        "checked_at": timestamp(),
    }
    try:
        import pyarrow
        import sklearn
        import torch
        import transformers
    except ImportError as exc:
        payload["error"] = repr(exc)
        write_json_atomic(payload, artifacts_dir / "environment.json", sort_keys=True)
        raise

    payload.update(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sklearn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    )
    write_json_atomic(payload, artifacts_dir / "environment.json", sort_keys=True)
    if args.device == "cuda" and not payload["cuda_available"]:
        raise RuntimeError("CUDA is required for this local run, but torch.cuda.is_available() is False.")
    if args.device == "cuda" and payload["gpu"] != "NVIDIA GeForce RTX 4070 Laptop GPU":
        raise RuntimeError(f"Expected NVIDIA GeForce RTX 4070 Laptop GPU, found {payload['gpu']!r}.")


def sample_command(args: argparse.Namespace, processed_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lucid_decoders.data.sample_ende",
        "--input",
        args.normalized_source,
        "--output",
        str(processed_dir / "all_trainable.jsonl"),
        "--train-per-label",
        str(args.train_per_label),
        "--validation-per-label",
        str(args.validation_per_label),
        "--test-per-label",
        str(args.test_per_label),
        "--seed",
        str(args.seed),
        "--summary-output",
        str(processed_dir / "sample_summary.json"),
    ]


def extract_command(args: argparse.Namespace, processed_dir: Path, artifacts_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "lucid_decoders.pipeline",
        "--stage",
        "extract-chunked",
        "--processed-dir",
        str(processed_dir),
        "--artifacts-dir",
        str(artifacts_dir),
        "--normalized-input",
        str(processed_dir / "all_trainable.jsonl"),
        "--model-name",
        args.model_name,
        "--source-lang",
        args.source_lang,
        "--target-lang",
        args.target_lang,
        "--device",
        args.device,
        "--chunk-size",
        str(args.chunk_size),
    ]


def train_command(
    args: argparse.Namespace,
    processed_dir: Path,
    artifacts_dir: Path,
    model_type: str,
    stage: str,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lucid_decoders.pipeline",
        "--stage",
        stage,
        "--processed-dir",
        str(processed_dir),
        "--artifacts-dir",
        str(artifacts_dir / model_type),
        "--model-type",
        model_type,
        "--seed",
        str(args.seed),
    ]
    if stage == "train-heads":
        command.extend(
            [
                "--min-train-examples",
                str(args.min_train_examples),
                "--head-train-jobs",
                str(args.head_train_jobs),
                "--persist-head-models",
                args.persist_head_models,
            ]
        )
    return command


def stage_outputs_complete(stage: str, processed_dir: Path, artifacts_dir: Path, args: argparse.Namespace) -> bool:
    if stage == "env-check":
        return False
    if stage == "sample":
        return (processed_dir / "all_trainable.jsonl").exists() and (processed_dir / "sample_summary.json").exists()
    if stage == "extract":
        return all(
            path.exists()
            for path in [
                processed_dir / "token_features.parquet",
                processed_dir / "sentence_features.parquet",
                processed_dir / "sentence_head_features.parquet",
                processed_dir / "mbart_extraction_report.json",
            ]
        )
    if stage.startswith("train-token:"):
        model_type = stage.split(":", 1)[1]
        return classifier_complete(artifacts_dir / model_type / "token_classifier", require_model=True)
    if stage.startswith("train-sentence:"):
        model_type = stage.split(":", 1)[1]
        return classifier_complete(artifacts_dir / model_type / "sentence_classifier", require_model=True)
    if stage.startswith("train-heads:"):
        model_type = stage.split(":", 1)[1]
        return head_classifier_complete(
            artifacts_dir / model_type / "sentence_head_classifier",
            require_best_model=args.persist_head_models in {"best", "all"},
        )
    if stage == "report":
        return False
    return False


def classifier_complete(path: Path, *, require_model: bool) -> bool:
    required = [path / "metrics.json", path / "test_predictions.parquet"]
    if require_model:
        required.append(path / "model.pkl")
    return all(item.exists() and item.stat().st_size > 0 for item in required)


def head_classifier_complete(path: Path, *, require_best_model: bool) -> bool:
    required = [path / "metrics.json", path / "head_metrics.csv", path / "test_predictions.parquet"]
    if require_best_model:
        required.extend([path / "best_model.pkl", path / "best_model_info.json"])
    return all(item.exists() and item.stat().st_size > 0 for item in required)


def collect_run_report(processed_dir: Path, artifacts_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "created_at": timestamp(),
        "sample": read_json_if_exists(processed_dir / "sample_summary.json"),
        "extraction": read_json_if_exists(processed_dir / "mbart_extraction_report.json"),
        "models": {},
    }
    for model_type in args.model_types:
        model_dir = artifacts_dir / model_type
        report["models"][model_type] = {
            "token_classifier": classifier_summary(model_dir / "token_classifier", "token_label", "token_pred"),
            "sentence_classifier": classifier_summary(model_dir / "sentence_classifier", "sentence_label", "sentence_pred"),
            "sentence_head_classifier": head_summary(model_dir / "sentence_head_classifier"),
        }
    write_json_atomic(report, artifacts_dir / "run_summary.json", sort_keys=True)
    return report


def classifier_summary(path: Path, label_col: str, pred_col: str) -> dict[str, Any]:
    metrics = read_json_if_exists(path / "metrics.json")
    predictions_path = path / "test_predictions.parquet"
    summary: dict[str, Any] = {"metrics": metrics, "prediction_rows": None, "test_confusion_matrix": None}
    if predictions_path.exists():
        frame = pd.read_parquet(predictions_path)
        summary["prediction_rows"] = int(len(frame))
        if label_col in frame.columns and pred_col in frame.columns:
            summary["test_confusion_matrix"] = confusion_matrix_dict(frame[label_col], frame[pred_col])
    return summary


def head_summary(path: Path) -> dict[str, Any]:
    metrics = read_json_if_exists(path / "metrics.json")
    predictions_path = path / "test_predictions.parquet"
    head_metrics_path = path / "head_metrics.csv"
    summary: dict[str, Any] = {
        "metrics": metrics,
        "prediction_rows": None,
        "best_head_test_confusion_matrix": None,
        "top_10_heads": [],
    }
    if predictions_path.exists():
        predictions = pd.read_parquet(predictions_path)
        summary["prediction_rows"] = int(len(predictions))
        best_head = (metrics or {}).get("best_head", {})
        if best_head:
            layer_id = int(best_head["layer_id"])
            head_id = int(best_head["head_id"])
            best_predictions = predictions[
                (predictions["layer_id"] == layer_id) & (predictions["head_id"] == head_id)
            ]
            summary["best_head_test_confusion_matrix"] = confusion_matrix_dict(
                best_predictions["sentence_label"],
                best_predictions["sentence_pred"],
            )
    if head_metrics_path.exists():
        head_metrics = pd.read_csv(head_metrics_path)
        summary["top_10_heads"] = head_metrics.head(10).round(6).to_dict(orient="records")
    return summary


def confusion_matrix_dict(labels: pd.Series, predictions: pd.Series) -> dict[str, int]:
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    return {
        "tn": int(((labels == 0) & (predictions == 0)).sum()),
        "fp": int(((labels == 0) & (predictions == 1)).sum()),
        "fn": int(((labels == 1) & (predictions == 0)).sum()),
        "tp": int(((labels == 1) & (predictions == 1)).sum()),
    }


def expand_stages(stages: list[str], model_types: list[str]) -> list[str]:
    if "all" not in stages:
        expanded: list[str] = []
        for stage in stages:
            if stage in TRAIN_KINDS:
                expanded.extend(f"{stage}:{model_type}" for model_type in model_types)
            else:
                expanded.append(stage)
        return expanded
    expanded = ["env-check", "sample", "extract"]
    for model_type in model_types:
        expanded.extend(
            [
                f"train-token:{model_type}",
                f"train-sentence:{model_type}",
                f"train-heads:{model_type}",
            ]
        )
    expanded.append("report")
    return expanded


def new_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_id": args.run_id,
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "config": {
            "processed_dir": args.processed_dir,
            "artifacts_dir": args.artifacts_dir,
            "normalized_source": args.normalized_source,
            "model_name": args.model_name,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "device": args.device,
            "chunk_size": args.chunk_size,
            "seed": args.seed,
            "counts": {
                "train_per_label": args.train_per_label,
                "validation_per_label": args.validation_per_label,
                "test_per_label": args.test_per_label,
            },
            "model_types": args.model_types,
            "head_train_jobs": args.head_train_jobs,
            "persist_head_models": args.persist_head_models,
        },
        "stages": {},
        "events": [],
    }


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any], path: Path) -> None:
    state["updated_at"] = timestamp()
    write_json_atomic(state, path, sort_keys=True)


def mark_stage(state: dict[str, Any], stage: str, status: str, message: str) -> None:
    entry = state["stages"].setdefault(stage, {})
    entry["status"] = status
    entry["message"] = message
    entry["updated_at"] = timestamp()
    if status == "running":
        entry["started_at"] = timestamp()
        entry["attempts"] = int(entry.get("attempts", 0)) + 1
    if status in {"completed", "skipped"}:
        entry["ended_at"] = timestamp()


def append_event(state: dict[str, Any], event_type: str, message: str) -> None:
    state.setdefault("events", []).append({"timestamp": timestamp(), "type": event_type, "message": message})


def acquire_lock(lock_path: Path, state: dict[str, Any], logs_dir: Path) -> None:
    if lock_path.exists():
        lock = read_json_if_exists(lock_path) or {}
        pid = lock.get("pid")
        if isinstance(pid, int) and is_process_running(pid):
            raise RuntimeError(f"Run lock is active for PID {pid}: {lock_path}")
        append_event(state, "stale_lock", f"Removed stale lock for PID {pid}.")
        append_run_log(logs_dir, f"Removed stale lock for PID `{pid}`.")
    write_json_atomic({"pid": os.getpid(), "created_at": timestamp()}, lock_path, sort_keys=True)


def release_lock(lock_path: Path) -> None:
    if lock_path.exists():
        lock = read_json_if_exists(lock_path) or {}
        if lock.get("pid") == os.getpid():
            lock_path.unlink()


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        synchronize = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def write_heartbeat(logs_dir: Path, state: dict[str, Any], active_stage: str, *, pid: int | None = None) -> None:
    payload = {
        "timestamp": timestamp(),
        "run_id": state.get("run_id"),
        "active_stage": active_stage,
        "pid": pid,
    }
    with (logs_dir / "heartbeat.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_stage_status(logs_dir: Path, state: dict[str, Any]) -> None:
    lines = ["# Stage Status", "", f"Updated: {timestamp()}", ""]
    for name, info in state.get("stages", {}).items():
        lines.append(f"- `{name}`: {info.get('status')} - {info.get('message')}")
    write_text_atomic("\n".join(lines) + "\n", logs_dir / "stage_status.md")


def append_run_log(logs_dir: Path, message: str) -> None:
    with (logs_dir / "RUN_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(f"- {timestamp()} - {message}\n")


def ensure_change_log(logs_dir: Path) -> None:
    path = logs_dir / "CHANGELOG.md"
    if not path.exists():
        write_text_atomic(
            "# Local Run Changelog\n\n"
            f"- {timestamp()} - Created local run log directory and recovery files.\n",
            path,
        )


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json", "path": str(path)}


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


if __name__ == "__main__":
    main()
