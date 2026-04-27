from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lucid_decoders.io import read_jsonl
from lucid_decoders.schemas import HallucinationSpan, TranslationExample

DEFAULT_ALIASES = {
    "example_id": ["example_id", "id", "segment_id", "row_id"],
    "source_text": ["source_text", "source", "src", "src_text", "context"],
    "hypothesis_text": [
        "hypothesis_text",
        "hypothesis",
        "translation",
        "generated_translation",
        "mt",
        "target",
    ],
    "sentence_label": [
        "sentence_label",
        "label",
        "hallucination_label",
        "is_hallucinated",
    ],
    "token_labels": [
        "token_labels",
        "hallucination_token_labels",
        "token_level_labels",
    ],
    "hallucination_spans": [
        "hallucination_spans",
        "spans",
        "hallucination_char_spans",
    ],
    "language_pair": ["language_pair", "lang_pair", "lp", "pair"],
}


@dataclass(slots=True)
class DatasetSpec:
    source_col: str | None = None
    target_col: str | None = None
    sentence_label_col: str | None = None
    token_label_col: str | None = None
    span_col: str | None = None
    id_col: str | None = None
    language_pair_col: str | None = None
    default_language_pair: str | None = None


def load_records(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(input_path)
    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "data" in payload:
            return list(payload["data"])
        raise ValueError(f"Unsupported JSON structure in {input_path}")
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(input_path, sep=sep).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {input_path}")


def load_wmt_examples(path: str | Path, spec: DatasetSpec | None = None) -> list[TranslationExample]:
    spec = spec or DatasetSpec()
    records = load_records(path)
    return [normalize_record(record, row_idx, spec) for row_idx, record in enumerate(records)]


def normalize_record(record: dict[str, Any], row_idx: int, spec: DatasetSpec) -> TranslationExample:
    source_col = spec.source_col or resolve_alias(record, DEFAULT_ALIASES["source_text"])
    target_col = spec.target_col or resolve_alias(record, DEFAULT_ALIASES["hypothesis_text"])
    id_col = spec.id_col or resolve_alias(record, DEFAULT_ALIASES["example_id"], required=False)
    label_col = spec.sentence_label_col or resolve_alias(
        record,
        DEFAULT_ALIASES["sentence_label"],
        required=False,
    )
    token_col = spec.token_label_col or resolve_alias(
        record,
        DEFAULT_ALIASES["token_labels"],
        required=False,
    )
    span_col = spec.span_col or resolve_alias(
        record,
        DEFAULT_ALIASES["hallucination_spans"],
        required=False,
    )
    pair_col = spec.language_pair_col or resolve_alias(
        record,
        DEFAULT_ALIASES["language_pair"],
        required=False,
    )

    token_labels = parse_int_list(record.get(token_col)) if token_col else None
    sentence_label = parse_int(record.get(label_col)) if label_col else None
    hallucination_spans = parse_spans(record.get(span_col)) if span_col else []

    if sentence_label is None and token_labels is not None:
        sentence_label = int(any(token_labels))
    if sentence_label is None and hallucination_spans:
        sentence_label = 1

    example_id = str(record.get(id_col) if id_col else row_idx)
    metadata = dict(record)

    return TranslationExample(
        example_id=example_id,
        source_text=str(record[source_col]),
        hypothesis_text=str(record[target_col]),
        sentence_label=sentence_label,
        token_labels=token_labels,
        hallucination_spans=hallucination_spans,
        language_pair=str(record[pair_col]) if pair_col else spec.default_language_pair,
        metadata=metadata,
    )


def resolve_alias(
    record: dict[str, Any],
    aliases: Iterable[str],
    required: bool = True,
) -> str | None:
    for alias in aliases:
        if alias in record:
            return alias
    if required:
        raise KeyError(f"Missing required column. Looked for aliases: {list(aliases)}")
    return None


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes"}:
        return 1
    if lowered in {"false", "no"}:
        return 0
    return int(float(lowered))


def parse_int_list(value: Any) -> list[int] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [parse_int(item) or 0 for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [parse_int(item) or 0 for item in payload]
        except json.JSONDecodeError:
            pass
        separator = "," if "," in text else " "
        return [parse_int(piece) or 0 for piece in text.split(separator) if piece.strip()]
    raise TypeError(f"Unsupported token label format: {type(value)!r}")


def parse_spans(value: Any) -> list[HallucinationSpan]:
    if value is None or value == "":
        return []
    payload = value
    if isinstance(value, str):
        payload = json.loads(value)
    spans: list[HallucinationSpan] = []
    for item in payload:
        if isinstance(item, dict):
            spans.append(
                HallucinationSpan(
                    start=int(item["start"]),
                    end=int(item["end"]),
                    label=int(item.get("label", 1)),
                )
            )
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            spans.append(HallucinationSpan(start=int(item[0]), end=int(item[1])))
            continue
        raise TypeError(f"Unsupported span format: {item!r}")
    return spans


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and normalize WMT hallucination datasets.")
    parser.add_argument("--input", required=True, help="Path to raw JSONL, JSON, CSV, or TSV input.")
    parser.add_argument("--source-col")
    parser.add_argument("--target-col")
    parser.add_argument("--sentence-label-col")
    parser.add_argument("--token-label-col")
    parser.add_argument("--span-col")
    parser.add_argument("--id-col")
    parser.add_argument("--language-pair-col")
    parser.add_argument("--language-pair")
    return parser

