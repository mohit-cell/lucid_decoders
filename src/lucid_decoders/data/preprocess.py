from __future__ import annotations

import argparse
import random
from collections import defaultdict

from lucid_decoders.config import SplitConfig
from lucid_decoders.data.wmt import DatasetSpec, load_wmt_examples
from lucid_decoders.io import write_jsonl
from lucid_decoders.schemas import TranslationExample


def assign_splits(
    examples: list[TranslationExample],
    config: SplitConfig,
) -> list[TranslationExample]:
    groups: dict[int | None, list[TranslationExample]] = defaultdict(list)
    if config.stratify:
        for example in examples:
            groups[example.sentence_label].append(example)
    else:
        groups[None] = list(examples)

    rng = random.Random(config.seed)
    assigned: list[TranslationExample] = []
    for group_examples in groups.values():
        rng.shuffle(group_examples)
        total = len(group_examples)
        train_cutoff = int(total * config.train_ratio)
        val_cutoff = int(total * (config.train_ratio + config.val_ratio))
        for idx, example in enumerate(group_examples):
            if idx < train_cutoff:
                example.split = "train"
            elif idx < val_cutoff:
                example.split = "validation"
            else:
                example.split = "test"
            assigned.append(example)
    return assigned


def filter_examples(examples: list[TranslationExample]) -> list[TranslationExample]:
    filtered: list[TranslationExample] = []
    for example in examples:
        if not example.source_text.strip():
            continue
        if not example.hypothesis_text.strip():
            continue
        filtered.append(example)
    return filtered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize and split WMT hallucination data.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-col")
    parser.add_argument("--target-col")
    parser.add_argument("--sentence-label-col")
    parser.add_argument("--token-label-col")
    parser.add_argument("--span-col")
    parser.add_argument("--id-col")
    parser.add_argument("--language-pair-col")
    parser.add_argument("--language-pair")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable label stratification when creating splits.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    spec = DatasetSpec(
        source_col=args.source_col,
        target_col=args.target_col,
        sentence_label_col=args.sentence_label_col,
        token_label_col=args.token_label_col,
        span_col=args.span_col,
        id_col=args.id_col,
        language_pair_col=args.language_pair_col,
        default_language_pair=args.language_pair,
    )
    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        stratify=not args.no_stratify,
    )

    examples = load_wmt_examples(args.input, spec)
    examples = filter_examples(examples)
    examples = assign_splits(examples, split_config)
    write_jsonl([example.to_dict() for example in examples], args.output)


if __name__ == "__main__":
    main()
