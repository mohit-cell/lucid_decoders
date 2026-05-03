from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


METRICS = ("entropy", "max", "variance", "topk_mass")
SUMMARY_STATS = ("mean", "std", "min", "max")
SENTENCE_STATS = ("mean", "max", "min", "std")


def attention_summary_columns(prefix: str) -> set[str]:
    columns: set[str] = set()
    for metric in METRICS:
        for stat in SUMMARY_STATS:
            columns.add(f"{prefix}_{metric}_{stat}")
    return columns


def sentence_aggregate_columns(token_columns: Iterable[str]) -> set[str]:
    columns: set[str] = set()
    for column in token_columns:
        for stat in SENTENCE_STATS:
            columns.add(f"{column}_{stat}")
    return columns


TOKEN_REQUIRED_COLUMNS = {
    "example_id",
    "language_pair",
    "split",
    "source_text",
    "hypothesis_text",
    "sentence_label",
    "token_label",
    "token_index",
    "token_text",
    "token_start_char",
    "token_end_char",
    "token_relative_position",
    "source_length_tokens",
    "target_length_tokens",
    "self_to_cross_max_ratio",
    "self_to_cross_entropy_ratio",
} | attention_summary_columns("cross") | attention_summary_columns("self")


SENTENCE_TOKEN_FEATURE_COLUMNS = (
    attention_summary_columns("cross")
    | attention_summary_columns("self")
    | {
        "token_relative_position",
        "source_length_tokens",
        "target_length_tokens",
        "self_to_cross_max_ratio",
        "self_to_cross_entropy_ratio",
    }
)


SENTENCE_REQUIRED_COLUMNS = {
    "example_id",
    "language_pair",
    "split",
    "source_text",
    "hypothesis_text",
    "sentence_label",
    "num_target_tokens",
} | sentence_aggregate_columns(SENTENCE_TOKEN_FEATURE_COLUMNS)


SENTENCE_HEAD_REQUIRED_COLUMNS = {
    "example_id",
    "language_pair",
    "split",
    "source_text",
    "hypothesis_text",
    "sentence_label",
    "layer_id",
    "head_id",
    "source_length_tokens",
    "target_length_tokens",
    "self_to_cross_max_ratio_mean",
    "self_to_cross_entropy_ratio_mean",
} | attention_summary_columns("cross") | attention_summary_columns("self")


FEATURE_CONTRACTS = {
    "token": TOKEN_REQUIRED_COLUMNS,
    "sentence": SENTENCE_REQUIRED_COLUMNS,
    "sentence_head": SENTENCE_HEAD_REQUIRED_COLUMNS,
}


def validate_feature_frame(frame: pd.DataFrame, kind: str) -> None:
    if kind not in FEATURE_CONTRACTS:
        raise ValueError(f"Unsupported feature contract kind: {kind}")
    if frame.empty:
        raise ValueError(f"{kind} feature frame is empty.")
    missing = sorted(FEATURE_CONTRACTS[kind] - set(frame.columns))
    if missing:
        raise ValueError(f"{kind} feature frame is missing required columns: {missing}")
