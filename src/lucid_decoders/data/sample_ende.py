from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from lucid_decoders.io import read_jsonl, write_json_atomic, write_jsonl_atomic


DEFAULT_COUNTS = {
    "train": 350,
    "validation": 75,
    "test": 75,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a balanced en-de JSONL subset for pipeline runs.")
    parser.add_argument("--input", default="data/processed/en_de/all_trainable.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-per-label", type=int, default=DEFAULT_COUNTS["train"])
    parser.add_argument("--validation-per-label", type=int, default=DEFAULT_COUNTS["validation"])
    parser.add_argument("--test-per-label", type=int, default=DEFAULT_COUNTS["test"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-source-chars", type=int)
    parser.add_argument("--max-target-chars", type=int)
    parser.add_argument(
        "--allow-positive-without-token-supervision",
        action="store_true",
        help="Allow positive sentence rows that do not include token_labels or hallucination_spans.",
    )
    parser.add_argument("--summary-output", help="Optional JSON summary path.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    selected, summary = sample_balanced_examples(
        records=read_jsonl(args.input),
        counts_per_label={
            "train": args.train_per_label,
            "validation": args.validation_per_label,
            "test": args.test_per_label,
        },
        seed=args.seed,
        max_source_chars=args.max_source_chars,
        max_target_chars=args.max_target_chars,
        require_positive_token_supervision=not args.allow_positive_without_token_supervision,
    )
    write_jsonl_atomic(selected, args.output)
    if args.summary_output:
        write_json_atomic(summary, args.summary_output)
    print(json.dumps(summary, indent=2))


def sample_balanced_examples(
    records: list[dict[str, Any]],
    counts_per_label: dict[str, int],
    seed: int,
    max_source_chars: int | None = None,
    max_target_chars: int | None = None,
    require_positive_token_supervision: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()

    for record in records:
        split = record.get("split")
        label = record.get("sentence_label")
        if split not in counts_per_label or label not in {0, 1}:
            skipped["unsupported_split_or_label"] += 1
            continue
        label = int(label)
        if max_source_chars is not None and len(record.get("source_text", "")) > max_source_chars:
            skipped["source_too_long"] += 1
            continue
        if max_target_chars is not None and len(record.get("hypothesis_text", "")) > max_target_chars:
            skipped["target_too_long"] += 1
            continue
        if (
            label == 1
            and require_positive_token_supervision
            and not has_token_supervision(record)
        ):
            skipped["positive_without_token_supervision"] += 1
            continue
        buckets[(split, label)].append(record)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    available: dict[str, int] = {}
    requested: dict[str, int] = {}
    for split, count in counts_per_label.items():
        for label in (0, 1):
            key = (split, label)
            requested[f"{split}:{label}"] = count
            available[f"{split}:{label}"] = len(buckets[key])
            if len(buckets[key]) < count:
                raise ValueError(
                    f"Not enough examples for split={split!r}, sentence_label={label}: "
                    f"requested {count}, available {len(buckets[key])}."
                )
            rng.shuffle(buckets[key])
            selected.extend(deepcopy(record) for record in buckets[key][:count])

    selected.sort(key=lambda row: (split_order(row.get("split")), int(row.get("sentence_label", 0)), row["example_id"]))
    rewritten_ids = make_example_ids_unique(selected)
    composition = Counter(
        f"{record.get('split')}:{int(record.get('sentence_label'))}:{has_token_supervision(record)}"
        for record in selected
    )
    summary = {
        "input_examples": len(records),
        "selected_examples": len(selected),
        "seed": seed,
        "requested_per_label": requested,
        "available_after_filters": available,
        "composition": dict(sorted(composition.items())),
        "skipped": dict(sorted(skipped.items())),
        "rewritten_duplicate_example_ids": rewritten_ids,
        "max_source_chars": max_source_chars,
        "max_target_chars": max_target_chars,
        "require_positive_token_supervision": require_positive_token_supervision,
    }
    return selected, summary


def has_token_supervision(record: dict[str, Any]) -> bool:
    return bool(record.get("token_labels") or record.get("hallucination_spans"))


def make_example_ids_unique(records: list[dict[str, Any]]) -> int:
    counts = Counter(str(record["example_id"]) for record in records)
    seen: Counter[str] = Counter()
    rewritten = 0
    for record in records:
        original_id = str(record["example_id"])
        if counts[original_id] <= 1:
            continue
        seen[original_id] += 1
        record["example_id"] = f"{original_id}__sample_{seen[original_id]}"
        metadata = dict(record.get("metadata", {}) or {})
        metadata["original_example_id"] = original_id
        record["metadata"] = metadata
        rewritten += 1
    return rewritten


def split_order(split: Any) -> int:
    return {"train": 0, "validation": 1, "test": 2}.get(str(split), 99)


if __name__ == "__main__":
    main()
