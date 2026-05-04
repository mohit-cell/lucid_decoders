from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
DEFAULT_SOURCE_LANG = "en_XX"
DEFAULT_TARGET_LANG = "de_DE"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Colab extraction progress and print the safe resume command."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root. Defaults to current directory.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed/en_de_full_features",
        help="Feature output directory used by extract-chunked.",
    )
    parser.add_argument(
        "--normalized-input",
        default="data/processed/en_de_full/all_trainable.jsonl",
        help="Normalized JSONL input used by extract-chunked.",
    )
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--source-lang", default=DEFAULT_SOURCE_LANG)
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    processed_dir = resolve_under_repo(repo_root, args.processed_dir)
    normalized_input = resolve_under_repo(repo_root, args.normalized_input)

    status = collect_status(
        repo_root=repo_root,
        processed_dir=processed_dir,
        normalized_input=normalized_input,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        device=args.device,
    )
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(format_status(status))


def resolve_under_repo(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def collect_status(
    *,
    repo_root: Path,
    processed_dir: Path,
    normalized_input: Path,
    chunk_size: int,
    model_name: str,
    source_lang: str,
    target_lang: str,
    device: str,
) -> dict[str, Any]:
    chunks_dir = processed_dir / "chunks"
    chunk_reports = read_chunk_reports(chunks_dir)
    final_report = read_json_if_exists(processed_dir / "mbart_extraction_report.json")
    merged_outputs = {
        "token": file_status(processed_dir / "token_features.parquet"),
        "sentence": file_status(processed_dir / "sentence_features.parquet"),
        "sentence_head": file_status(processed_dir / "sentence_head_features.parquet"),
        "report": file_status(processed_dir / "mbart_extraction_report.json"),
    }
    completed_reports = [report for report in chunk_reports if report.get("status") == "completed"]
    chunk_summary = {
        "chunks_dir": str(chunks_dir),
        "exists": chunks_dir.exists(),
        "report_files": len(chunk_reports),
        "completed_chunks": len(completed_reports),
        "processed_examples": sum_int(completed_reports, "processed_examples"),
        "skipped_examples": sum_int(completed_reports, "skipped_examples"),
        "token_rows": sum_int(completed_reports, "token_rows"),
        "sentence_rows": sum_int(completed_reports, "sentence_rows"),
        "sentence_head_rows": sum_int(completed_reports, "sentence_head_rows"),
    }
    return {
        "repo_root": str(repo_root),
        "repo_exists": repo_root.exists(),
        "drive_backed": is_drive_backed(repo_root),
        "normalized_input": {
            "path": str(normalized_input),
            "exists": normalized_input.exists(),
            "line_count": count_lines(normalized_input) if normalized_input.exists() else None,
        },
        "processed_dir": {
            "path": str(processed_dir),
            "exists": processed_dir.exists(),
        },
        "chunks": chunk_summary,
        "merged_outputs": merged_outputs,
        "final_report": final_report,
        "resume_command": build_resume_command(
            normalized_input=normalized_input,
            processed_dir=processed_dir,
            model_name=model_name,
            source_lang=source_lang,
            target_lang=target_lang,
            device=device,
            chunk_size=chunk_size,
        ),
    }


def read_chunk_reports(chunks_dir: Path) -> list[dict[str, Any]]:
    if not chunks_dir.exists():
        return []
    reports: list[dict[str, Any]] = []
    for report_path in sorted(chunks_dir.glob("*.report.json")):
        report = read_json_if_exists(report_path)
        if report is None:
            reports.append({"path": str(report_path), "status": "unreadable"})
        else:
            report["path"] = str(report_path)
            reports.append(report)
    return reports


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"path": str(path), "status": "invalid_json"}


def file_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "size": human_size(path.stat().st_size) if path.exists() else None,
    }


def count_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def sum_int(reports: list[dict[str, Any]], key: str) -> int:
    total = 0
    for report in reports:
        value = report.get(key, 0)
        if isinstance(value, int):
            total += value
    return total


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{size_bytes} B"


def is_drive_backed(path: Path) -> bool:
    return str(path).startswith("/content/drive/")


def build_resume_command(
    *,
    normalized_input: Path,
    processed_dir: Path,
    model_name: str,
    source_lang: str,
    target_lang: str,
    device: str,
    chunk_size: int,
) -> str:
    return "\n".join(
        [
            "!PYTHONPATH=src python -m lucid_decoders.pipeline \\",
            "  --stage extract-chunked \\",
            f"  --normalized-input {normalized_input} \\",
            f"  --processed-dir {processed_dir} \\",
            f"  --model-name {model_name} \\",
            f"  --source-lang {source_lang} \\",
            f"  --target-lang {target_lang} \\",
            f"  --device {device} \\",
            f"  --chunk-size {chunk_size}",
        ]
    )


def format_status(status: dict[str, Any]) -> str:
    chunks = status["chunks"]
    normalized_input = status["normalized_input"]
    outputs = status["merged_outputs"]
    final_report = status["final_report"] or {}
    lines = [
        "Colab recovery status",
        f"- repo_root: {status['repo_root']}",
        f"- repo_exists: {status['repo_exists']}",
        f"- drive_backed: {status['drive_backed']}",
        f"- normalized_input: {normalized_input['path']}",
        f"- normalized_input_exists: {normalized_input['exists']}",
        f"- normalized_input_lines: {normalized_input['line_count']}",
        f"- processed_dir: {status['processed_dir']['path']}",
        f"- processed_dir_exists: {status['processed_dir']['exists']}",
        f"- chunks_dir_exists: {chunks['exists']}",
        f"- chunk_report_files: {chunks['report_files']}",
        f"- completed_chunks: {chunks['completed_chunks']}",
        f"- processed_examples_from_chunks: {chunks['processed_examples']}",
        f"- skipped_examples_from_chunks: {chunks['skipped_examples']}",
        f"- token_rows_from_chunks: {chunks['token_rows']}",
        f"- sentence_rows_from_chunks: {chunks['sentence_rows']}",
        f"- sentence_head_rows_from_chunks: {chunks['sentence_head_rows']}",
        f"- final_report_processed_examples: {final_report.get('processed_examples')}",
        f"- final_report_completed_chunks: {final_report.get('completed_chunks')}",
        "",
        "Merged outputs",
    ]
    for name, output in outputs.items():
        lines.append(f"- {name}: exists={output['exists']} size={output['size']}")
    lines.extend(["", "Resume command", status["resume_command"]])
    return "\n".join(lines)


if __name__ == "__main__":
    main()

