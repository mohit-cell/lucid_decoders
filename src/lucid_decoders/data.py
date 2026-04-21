"""Data loading utilities for MT hallucination examples."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


SOURCE_ALIASES = ("source", "src", "source_sentence", "src_text", "context")
GENERATED_ALIASES = (
    "generated",
    "translation",
    "hypothesis",
    "mt",
    "target",
    "generated_translation",
    "gen_text",
)
LABEL_ALIASES = ("label", "hallucination", "is_hallucination", "hallucinated", "binary_label")
ID_ALIASES = ("id", "example_id", "doc_id", "segment_id")

TRUE_LABELS = {"1", "true", "yes", "y", "hallucinated", "hallucination", "positive", "bad"}
FALSE_LABELS = {"0", "false", "no", "n", "not_hallucinated", "faithful", "negative", "ok", "good"}


@dataclass(frozen=True)
class MTExample:
    """A source sentence, generated translation, and optional hallucination labels."""

    example_id: str
    source: str
    generated: str
    label: int | None = None
    source_lang: str | None = None
    target_lang: str | None = None
    token_labels: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_label(value: Any) -> int:
    """Parse a binary hallucination label into 0 or 1."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"Expected binary integer label, got {value!r}.")
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return int(value)
        raise ValueError(f"Expected binary float label, got {value!r}.")

    text = str(value).strip().lower()
    if text in TRUE_LABELS:
        return 1
    if text in FALSE_LABELS:
        return 0

    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse binary label {value!r}.") from exc
    if numeric in (0.0, 1.0):
        return int(numeric)
    raise ValueError(f"Expected binary numeric label, got {value!r}.")


def load_examples(
    path: str | Path | None = None,
    *,
    hf_dataset: str | None = None,
    split: str = "train",
    source_col: str = "source",
    generated_col: str = "generated",
    label_col: str | None = "label",
    id_col: str | None = None,
    source_lang_col: str | None = None,
    target_lang_col: str | None = None,
    token_labels_col: str | None = None,
    default_source_lang: str | None = None,
    default_target_lang: str | None = None,
) -> list[MTExample]:
    """Load hallucination examples from a local file or a Hugging Face dataset.

    Local files may be CSV, JSON, or JSONL. Hugging Face dataset loading is kept
    configurable because WMT hallucination releases and mirrors differ in schema.
    """

    if path is None and hf_dataset is None:
        raise ValueError("Provide either `path` or `hf_dataset`.")
    if path is not None and hf_dataset is not None:
        raise ValueError("Provide only one of `path` or `hf_dataset`.")

    rows = _read_local_rows(Path(path)) if path is not None else _read_hf_rows(hf_dataset, split)
    examples = [
        _row_to_example(
            row,
            index=i,
            source_col=source_col,
            generated_col=generated_col,
            label_col=label_col,
            id_col=id_col,
            source_lang_col=source_lang_col,
            target_lang_col=target_lang_col,
            token_labels_col=token_labels_col,
            default_source_lang=default_source_lang,
            default_target_lang=default_target_lang,
        )
        for i, row in enumerate(rows)
    ]
    return clean_examples(examples)


def clean_examples(examples: Iterable[MTExample]) -> list[MTExample]:
    """Drop examples with empty source or generated text."""

    cleaned: list[MTExample] = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        source = example.source.strip()
        generated = example.generated.strip()
        if not source or not generated:
            continue
        key = (source, generated)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            MTExample(
                example_id=example.example_id,
                source=source,
                generated=generated,
                label=example.label,
                source_lang=example.source_lang,
                target_lang=example.target_lang,
                token_labels=example.token_labels,
                metadata=example.metadata,
            )
        )
    return cleaned


def _read_local_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "examples", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
        raise ValueError(f"JSON file {path} must contain a list or a data/examples/rows list.")
    raise ValueError(f"Unsupported data file extension: {path.suffix}")


def _read_hf_rows(hf_dataset: str | None, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the `ml` extra to load Hugging Face datasets.") from exc
    dataset = load_dataset(hf_dataset, split=split)
    return [dict(row) for row in dataset]


def _row_to_example(
    row: dict[str, Any],
    *,
    index: int,
    source_col: str,
    generated_col: str,
    label_col: str | None,
    id_col: str | None,
    source_lang_col: str | None,
    target_lang_col: str | None,
    token_labels_col: str | None,
    default_source_lang: str | None,
    default_target_lang: str | None,
) -> MTExample:
    source = _get_value(row, source_col, SOURCE_ALIASES)
    generated = _get_value(row, generated_col, GENERATED_ALIASES)
    label_value = _get_optional_value(row, label_col, LABEL_ALIASES) if label_col else None
    example_id = str(_get_optional_value(row, id_col, ID_ALIASES) or index)
    source_lang = _get_optional_value(row, source_lang_col, ()) or default_source_lang
    target_lang = _get_optional_value(row, target_lang_col, ()) or default_target_lang
    token_labels = _parse_token_labels(_get_optional_value(row, token_labels_col, ()))

    metadata = {k: v for k, v in row.items() if k not in {source_col, generated_col, label_col, id_col}}
    return MTExample(
        example_id=example_id,
        source=str(source),
        generated=str(generated),
        label=parse_label(label_value) if label_value is not None else None,
        source_lang=str(source_lang) if source_lang is not None else None,
        target_lang=str(target_lang) if target_lang is not None else None,
        token_labels=token_labels,
        metadata=metadata,
    )


def _get_value(row: dict[str, Any], preferred: str | None, aliases: Iterable[str]) -> Any:
    value = _get_optional_value(row, preferred, aliases)
    if value is None:
        candidates = ", ".join([c for c in [preferred, *aliases] if c])
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return value


def _get_optional_value(row: dict[str, Any], preferred: str | None, aliases: Iterable[str]) -> Any:
    if preferred and preferred in row and row[preferred] not in (None, ""):
        return row[preferred]
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    return None


def _parse_token_labels(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = value.split()
    if not isinstance(value, list):
        raise ValueError("Token labels must be a list, JSON list, or whitespace-separated labels.")
    return [parse_label(item) for item in value]

