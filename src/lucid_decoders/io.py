from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_table(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix == ".tsv":
        return pd.read_csv(input_path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    if suffix == ".jsonl":
        return pd.DataFrame(read_jsonl(input_path))
    raise ValueError(f"Unsupported table format: {input_path}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    output_path = ensure_parent_dir(path)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(output_path, index=False)
        return
    if suffix == ".tsv":
        frame.to_csv(output_path, sep="\t", index=False)
        return
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(output_path, index=False)
        return
    if suffix == ".jsonl":
        write_jsonl(frame.to_dict(orient="records"), output_path)
        return
    raise ValueError(f"Unsupported table format: {output_path}")

