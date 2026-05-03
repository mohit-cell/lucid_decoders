from __future__ import annotations

import argparse
import csv
import json
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from lucid_decoders.data.validate_ende import raise_for_missing_data, validate_wmt_roots
from lucid_decoders.io import write_jsonl
from lucid_decoders.schemas import HallucinationSpan, TranslationExample


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare normalized en-de datasets from the cloned WMT22/WMT23 repos."
    )
    parser.add_argument("--wmt22-root", default="data/raw/wmt22")
    parser.add_argument("--wmt23-root", default="data/raw/wmt23")
    parser.add_argument("--output-dir", default="data/processed/en_de")
    parser.add_argument(
        "--mqm-threshold",
        type=float,
        default=0.0,
        help="Binary threshold for WMT22 sentence MQM z-score. label=1 if mean_zscore < threshold.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the raw-file validation check before preparing data.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.skip_validation:
        raise_for_missing_data(validate_wmt_roots(args.wmt22_root, args.wmt23_root))
    summary = prepare_ende_datasets(
        wmt22_root=args.wmt22_root,
        wmt23_root=args.wmt23_root,
        output_dir=args.output_dir,
        mqm_threshold=args.mqm_threshold,
    )
    print(json.dumps(summary, indent=2))


def prepare_ende_datasets(
    wmt22_root: str | Path,
    wmt23_root: str | Path,
    output_dir: str | Path,
    mqm_threshold: float = 0.0,
) -> dict[str, Any]:
    wmt22_root = Path(wmt22_root)
    wmt23_root = Path(wmt23_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_examples = build_wmt22_sentence_examples(wmt22_root, mqm_threshold=mqm_threshold)
    token_examples = build_wmt22_word_examples(wmt22_root)
    span_examples = build_wmt23_task2_examples(wmt23_root)
    hallucination_examples = build_wmt23_hallucination_gold_examples(wmt23_root)
    all_examples = sentence_examples + token_examples + span_examples + hallucination_examples
    trainable_examples = [example for example in all_examples if is_trainable_for_attention(example)]

    write_jsonl([example.to_dict() for example in sentence_examples], output_dir / "wmt22_ende_sentence_mqm.jsonl")
    write_jsonl([example.to_dict() for example in token_examples], output_dir / "wmt22_ende_word_mqm.jsonl")
    write_jsonl([example.to_dict() for example in span_examples], output_dir / "wmt23_ende_task2.jsonl")
    write_jsonl(
        [example.to_dict() for example in hallucination_examples],
        output_dir / "wmt23_ende_hallucination_gold.jsonl",
    )
    write_jsonl([example.to_dict() for example in all_examples], output_dir / "all_examples.jsonl")
    write_jsonl([example.to_dict() for example in trainable_examples], output_dir / "all_trainable.jsonl")

    summary = {
        "wmt22_sentence_examples": len(sentence_examples),
        "wmt22_word_examples": len(token_examples),
        "wmt23_task2_examples": len(span_examples),
        "wmt23_hallucination_gold_examples": len(hallucination_examples),
        "all_examples": len(all_examples),
        "all_trainable_examples": len(trainable_examples),
        "output_dir": str(output_dir),
        "outputs": {
            "all_examples": str(output_dir / "all_examples.jsonl"),
            "all_trainable": str(output_dir / "all_trainable.jsonl"),
            "wmt22_sentence_mqm": str(output_dir / "wmt22_ende_sentence_mqm.jsonl"),
            "wmt22_word_mqm": str(output_dir / "wmt22_ende_word_mqm.jsonl"),
            "wmt23_task2": str(output_dir / "wmt23_ende_task2.jsonl"),
            "wmt23_hallucination_gold": str(output_dir / "wmt23_ende_hallucination_gold.jsonl"),
        },
        "notes": [
            "WMT23 hallucination gold examples do not include source sentences in the repo. "
            "They are exported for inspection with empty source_text and missing_source=true, "
            "but excluded from all_trainable.jsonl."
        ],
    }
    return summary


def is_trainable_for_attention(example: TranslationExample) -> bool:
    if not example.source_text.strip():
        return False
    if not example.hypothesis_text.strip():
        return False
    if example.sentence_label is None:
        return False
    return True


def build_wmt22_sentence_examples(
    wmt22_root: Path,
    mqm_threshold: float,
) -> list[TranslationExample]:
    sentence_specs = [
        (
            wmt22_root / "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2020.csv",
            "train",
            "2020",
        ),
        (
            wmt22_root / "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2021-ted.csv",
            "train",
            "2021-ted",
        ),
        (
            wmt22_root / "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2021.csv",
            "train",
            "2021",
        ),
        (
            wmt22_root / "train-dev_data/task1_mqm/dev/en-de/en-de-mqm.2022_dev.csv",
            "validation",
            "2022-dev",
        ),
    ]
    examples: list[TranslationExample] = []
    for csv_path, split_name, corpus_name in sentence_specs:
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if frame.columns[0] in {"", "Unnamed: 0"}:
            frame = frame.rename(columns={frame.columns[0]: "row_index"})
        if "zscore" not in frame.columns and "z_score" in frame.columns:
            frame = frame.rename(columns={"z_score": "zscore"})
        if "system" not in frame.columns:
            frame["system"] = "gold"
        if "doc" not in frame.columns:
            frame["doc"] = corpus_name
        if "doc_id" not in frame.columns:
            frame["doc_id"] = frame["doc"]
        if "rater" not in frame.columns:
            if "annotator" in frame.columns:
                frame = frame.rename(columns={"annotator": "rater"})
            else:
                frame["rater"] = "unknown"

        group_cols = ["system", "doc", "doc_id", "seg_id", "src", "mt"]
        aggregated = (
            frame.groupby(group_cols, dropna=False)
            .agg(
                score_mean=("score", "mean"),
                zscore_mean=("zscore", "mean"),
                annotator_count=("rater", "nunique"),
            )
            .reset_index()
        )
        for _, row in aggregated.iterrows():
            zscore_mean = float(row["zscore_mean"])
            sentence_label = int(zscore_mean < mqm_threshold)
            example_id = (
                f"wmt22-sent-en-de-{corpus_name}-"
                f"{safe_id(row['system'])}-{safe_id(row['doc'])}-{safe_id(row['seg_id'])}"
            )
            examples.append(
                TranslationExample(
                    example_id=example_id,
                    source_text=str(row["src"]),
                    hypothesis_text=str(row["mt"]),
                    sentence_label=sentence_label,
                    language_pair="en-de",
                    split=split_name,
                    metadata={
                        "dataset": "wmt22_task1_mqm",
                        "corpus": corpus_name,
                        "system": row["system"],
                        "doc": row["doc"],
                        "doc_id": row["doc_id"],
                        "seg_id": row["seg_id"],
                        "score_mean": float(row["score_mean"]),
                        "zscore_mean": zscore_mean,
                        "annotator_count": int(row["annotator_count"]),
                        "label_rule": f"mean_zscore < {mqm_threshold}",
                    },
                )
            )
    return examples


def build_wmt22_word_examples(wmt22_root: Path) -> list[TranslationExample]:
    archives = [
        (
            wmt22_root / "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2020.tar.gz",
            "train",
            "2020",
        ),
        (
            wmt22_root / "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2021-news.tar.gz",
            "train",
            "2021-news",
        ),
        (
            wmt22_root / "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2021-ted.tar.gz",
            "train",
            "2021-ted",
        ),
        (
            wmt22_root / "train-dev_data/task1_word-level/dev/en-de_mqm/en-de-dev-2022.zip",
            "validation",
            "2022-dev",
        ),
    ]
    examples: list[TranslationExample] = []
    for archive_path, split_name, corpus_name in archives:
        if not archive_path.exists():
            continue
        if archive_path.suffix == ".zip":
            records = read_parallel_files_from_zip(archive_path)
        else:
            records = read_parallel_files_from_tar(archive_path)

        for idx, record in enumerate(records):
            source_text = record["source_text"]
            raw_target_tokens = record["target_text"].split()
            target_tokens = strip_eos(raw_target_tokens)
            tag_tokens = record["tag_text"].split()
            if raw_target_tokens and raw_target_tokens[-1] == "<EOS>" and len(tag_tokens) == len(target_tokens) + 1:
                tag_tokens = tag_tokens[:-1]
            if len(target_tokens) != len(tag_tokens):
                raise ValueError(
                    f"Token/tag length mismatch in {archive_path} row {idx}: "
                    f"{len(target_tokens)} tokens vs {len(tag_tokens)} tags."
                )
            hypothesis_text, token_spans = tokens_to_text_and_spans(target_tokens)
            hallucination_spans = [
                HallucinationSpan(start=start, end=end)
                for (start, end), tag in zip(token_spans, tag_tokens)
                if normalize_tag(tag) == 1
            ]
            sentence_label = int(any(tag == 1 for tag in map(normalize_tag, tag_tokens)))
            example_id = f"wmt22-word-en-de-{corpus_name}-{idx}"
            examples.append(
                TranslationExample(
                    example_id=example_id,
                    source_text=source_text,
                    hypothesis_text=hypothesis_text,
                    sentence_label=sentence_label,
                    hallucination_spans=merge_spans(hallucination_spans),
                    language_pair="en-de",
                    split=split_name,
                    metadata={
                        "dataset": "wmt22_task1_word_level_mqm",
                        "corpus": corpus_name,
                        "whitespace_tokens": target_tokens,
                        "word_tags": tag_tokens,
                    },
                )
            )
    return examples


def build_wmt23_task2_examples(wmt23_root: Path) -> list[TranslationExample]:
    file_specs = [
        (wmt23_root / "task_2/train/2020_en-de_processed.tsv", "train", "2020"),
        (wmt23_root / "task_2/train/2021_TED_en-de_processed.tsv", "train", "2021-ted"),
        (wmt23_root / "task_2/train/2021_en-de_processed.tsv", "train", "2021"),
        (wmt23_root / "task_2/dev/2022_en-de_dev_processed.tsv", "validation", "2022-dev"),
        (wmt23_root / "task_2/dev/2022_en-de_test_processed.tsv", "test", "2022-test"),
    ]
    examples: list[TranslationExample] = []
    for path, split_name, corpus_name in file_specs:
        if not path.exists():
            continue
        grouped_rows: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
            for parts in reader:
                if not parts or not any(part.strip() for part in parts):
                    continue
                if len(parts) == 10 and not parts[-1].strip():
                    parts = parts[:-1]
                if len(parts) != 9:
                    raise ValueError(f"Unexpected column count in {path}: {len(parts)} -> {parts[:5]}")
                if parts[0] == "mt_model" or parts[6] == "start_ids":
                    continue
                row = {
                    "mt_model": parts[0],
                    "doc_id": parts[1],
                    "seg_id": parts[2],
                    "annotator_id": parts[3],
                    "source": parts[4],
                    "target": parts[5],
                    "start_ids": parts[6],
                    "end_ids": parts[7],
                    "error_types": parts[8],
                }
                key = (row["mt_model"], row["doc_id"], row["seg_id"], row["source"], row["target"])
                grouped_rows[key].append(row)

        for key, rows in grouped_rows.items():
            mt_model, doc_id, seg_id, source_text, target_text = key
            spans: list[HallucinationSpan] = []
            error_types: list[str] = []
            for row in rows:
                row_spans, row_types = parse_task2_span_fields(
                    row["start_ids"],
                    row["end_ids"],
                    row["error_types"],
                )
                spans.extend(row_spans)
                error_types.extend(row_types)

            merged = merge_spans(spans)
            sentence_label = int(bool(merged))
            example_id = f"wmt23-task2-en-de-{corpus_name}-{safe_id(mt_model)}-{safe_id(doc_id)}-{safe_id(seg_id)}"
            examples.append(
                TranslationExample(
                    example_id=example_id,
                    source_text=source_text,
                    hypothesis_text=target_text,
                    sentence_label=sentence_label,
                    hallucination_spans=merged,
                    language_pair="en-de",
                    split=split_name,
                    metadata={
                        "dataset": "wmt23_task2",
                        "corpus": corpus_name,
                        "mt_model": mt_model,
                        "doc_id": doc_id,
                        "seg_id": seg_id,
                        "annotator_count": len(rows),
                        "error_types": sorted(set(error_types)),
                    },
                )
            )
    return examples


def build_wmt23_hallucination_gold_examples(wmt23_root: Path) -> list[TranslationExample]:
    t1s_path = wmt23_root / "gold_labels/hallucinations_gold_T1s.tsv"
    t1w_path = wmt23_root / "gold_labels/hallucinations_gold_T1w.tsv"
    t2_path = wmt23_root / "gold_labels/hallucinations_gold_T2.tsv"

    sentence_scores: dict[int, str] = {}
    with t1s_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for lp, gold, sid, score in reader:
            if lp != "en-de":
                continue
            sentence_scores[int(sid)] = score

    word_rows: dict[int, dict[str, str]] = {}
    with t1w_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for lp, gold, sid, mt, mttok, tags in reader:
            if lp != "en-de":
                continue
            word_rows[int(sid)] = {
                "mt": mt,
                "mttok": mttok,
                "tags": tags,
            }

    span_rows: dict[int, dict[str, str]] = {}
    with t2_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for lp, gold, sid, mt, start_ids, end_ids, error in reader:
            if lp != "en-de":
                continue
            span_rows[int(sid)] = {
                "mt": mt,
                "start_ids": start_ids,
                "end_ids": end_ids,
                "error": error,
            }

    examples: list[TranslationExample] = []
    for sid in sorted(set(sentence_scores) | set(word_rows) | set(span_rows)):
        word_row = word_rows.get(sid, {})
        span_row = span_rows.get(sid, {})
        hypothesis_text = word_row.get("mt") or span_row.get("mt") or ""

        hallucination_spans: list[HallucinationSpan] = []
        if span_row:
            hallucination_spans, error_types = parse_task2_span_fields(
                span_row["start_ids"],
                span_row["end_ids"],
                span_row["error"],
            )
            hallucination_spans = merge_spans(hallucination_spans)
        else:
            error_types = []

        score_value = sentence_scores.get(sid)
        sentence_label = 1 if score_value == "hallucination" else int(bool(hallucination_spans))
        examples.append(
            TranslationExample(
                example_id=f"wmt23-hallucination-gold-en-de-{sid}",
                source_text="",
                hypothesis_text=hypothesis_text,
                sentence_label=sentence_label,
                hallucination_spans=hallucination_spans,
                language_pair="en-de",
                split="test",
                metadata={
                    "dataset": "wmt23_hallucination_gold",
                    "sid": sid,
                    "score": score_value,
                    "error_types": error_types,
                    "missing_source": True,
                    "tokenized_target": word_row.get("mttok"),
                    "word_tags": word_row.get("tags"),
                },
            )
        )
    return examples


def read_parallel_files_from_tar(path: Path) -> list[dict[str, str]]:
    with tarfile.open(path, "r:gz") as archive:
        src_member = find_archive_member(archive.getnames(), [".src"])
        mt_member = find_archive_member(archive.getnames(), [".mt"])
        tags_member = find_archive_member(archive.getnames(), [".tags", ".word_level.tags"])
        src_lines = read_member_lines_tar(archive, src_member)
        mt_lines = read_member_lines_tar(archive, mt_member)
        tag_lines = read_member_lines_tar(archive, tags_member)
    return build_parallel_records(src_lines, mt_lines, tag_lines)


def read_parallel_files_from_zip(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not is_archive_junk(name) and not name.endswith("/")]
        src_member = find_archive_member(names, [".src"])
        mt_member = find_archive_member(names, [".mt"])
        tags_member = find_archive_member(names, [".tags", ".word_level.tags"])
        src_lines = read_member_lines_zip(archive, src_member)
        mt_lines = read_member_lines_zip(archive, mt_member)
        tag_lines = read_member_lines_zip(archive, tags_member)
    return build_parallel_records(src_lines, mt_lines, tag_lines)


def read_member_lines_tar(archive: tarfile.TarFile, member_name: str) -> list[str]:
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(f"Unable to read {member_name} from archive.")
    return [line.decode("utf-8").rstrip("\n") for line in extracted]


def read_member_lines_zip(archive: zipfile.ZipFile, member_name: str) -> list[str]:
    with archive.open(member_name) as handle:
        return [line.decode("utf-8").rstrip("\n") for line in handle]


def find_archive_member(names: Iterable[str], suffixes: list[str]) -> str:
    candidates = [
        name
        for name in names
        if not is_archive_junk(name)
        and any(name.endswith(suffix) for suffix in suffixes)
    ]
    if not candidates:
        raise FileNotFoundError(f"Could not find archive member with suffixes {suffixes}")
    candidates.sort()
    return candidates[0]


def is_archive_junk(name: str) -> bool:
    parts = [part for part in Path(name).parts if part not in {".", ""}]
    return any(part.startswith("._") for part in parts) or name.startswith("__MACOSX/")


def build_parallel_records(
    src_lines: list[str],
    mt_lines: list[str],
    tag_lines: list[str],
) -> list[dict[str, str]]:
    if not (len(src_lines) == len(mt_lines) == len(tag_lines)):
        raise ValueError(
            f"Parallel file length mismatch: {len(src_lines)} src, {len(mt_lines)} mt, {len(tag_lines)} tags"
        )
    return [
        {
            "source_text": src,
            "target_text": mt,
            "tag_text": tags,
        }
        for src, mt, tags in zip(src_lines, mt_lines, tag_lines)
    ]


def strip_eos(tokens: list[str]) -> list[str]:
    if tokens and tokens[-1] == "<EOS>":
        return tokens[:-1]
    return tokens


def normalize_tag(tag: str) -> int:
    return 1 if tag.strip().upper() == "BAD" else 0


def tokens_to_text_and_spans(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append((start, cursor))
    return "".join(pieces), spans


def parse_task2_span_fields(
    start_ids: str,
    end_ids: str,
    error_types: str,
) -> tuple[list[HallucinationSpan], list[str]]:
    if start_ids == "-1" or end_ids == "-1" or error_types == "no-error":
        return [], []
    starts = [int(value) for value in start_ids.split()]
    ends = [int(value) for value in end_ids.split()]
    types = error_types.split()
    if not (len(starts) == len(ends) == len(types)):
        raise ValueError(
            "Task2 span field mismatch: "
            f"{len(starts)} starts, {len(ends)} ends, {len(types)} types"
        )
    spans = [HallucinationSpan(start=start, end=end + 1) for start, end in zip(starts, ends)]
    return spans, types


def merge_spans(spans: list[HallucinationSpan]) -> list[HallucinationSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    merged: list[HallucinationSpan] = [HallucinationSpan(start=ordered[0].start, end=ordered[0].end)]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start <= last.end:
            last.end = max(last.end, span.end)
        else:
            merged.append(HallucinationSpan(start=span.start, end=span.end))
    return merged


def safe_id(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
