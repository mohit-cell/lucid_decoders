from __future__ import annotations

from typing import Any

import pandas as pd


SENTENCE_EXCLUDE = {
    "token_index",
    "token_start_char",
    "token_end_char",
}


def build_sentence_feature_frame(token_frame: pd.DataFrame) -> pd.DataFrame:
    if token_frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    numeric_columns = [
        col
        for col in token_frame.columns
        if pd.api.types.is_numeric_dtype(token_frame[col])
        and col not in {"sentence_label", "token_label"}
        and col not in SENTENCE_EXCLUDE
    ]

    for example_id, group in token_frame.groupby("example_id", sort=False):
        row: dict[str, Any] = {
            "example_id": example_id,
            "language_pair": group["language_pair"].iloc[0],
            "split": group["split"].iloc[0],
            "source_text": group["source_text"].iloc[0],
            "hypothesis_text": group["hypothesis_text"].iloc[0],
            "sentence_label": group["sentence_label"].iloc[0],
            "num_target_tokens": int(len(group)),
        }
        for column in numeric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_max"] = float(group[column].max())
            row[f"{column}_min"] = float(group[column].min())
            row[f"{column}_std"] = float(group[column].std(ddof=0))

        if "token_score" in group.columns:
            row["token_score_mean"] = float(group["token_score"].mean())
            row["token_score_max"] = float(group["token_score"].max())
            row["token_score_fraction_ge_0_5"] = float((group["token_score"] >= 0.5).mean())
        rows.append(row)

    return pd.DataFrame(rows)

