from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class RequiredPath:
    dataset: str
    purpose: str
    relative_path: str


@dataclass(slots=True)
class DataValidationIssue:
    dataset: str
    purpose: str
    path: str


@dataclass(slots=True)
class DataValidationReport:
    ok: bool
    wmt22_root: str
    wmt23_root: str
    checked_paths: int
    missing: list[DataValidationIssue]

    def to_dict(self) -> dict:
        return asdict(self)


WMT22_REQUIRED_PATHS = [
    RequiredPath(
        "wmt22",
        "sentence_mqm_train_2020",
        "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2020.csv",
    ),
    RequiredPath(
        "wmt22",
        "sentence_mqm_train_2021_ted",
        "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2021-ted.csv",
    ),
    RequiredPath(
        "wmt22",
        "sentence_mqm_train_2021_news",
        "train-dev_data/task1_mqm/train/en-de/en-de-mqm.2021.csv",
    ),
    RequiredPath(
        "wmt22",
        "sentence_mqm_dev_2022",
        "train-dev_data/task1_mqm/dev/en-de/en-de-mqm.2022_dev.csv",
    ),
    RequiredPath(
        "wmt22",
        "word_mqm_train_2020",
        "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2020.tar.gz",
    ),
    RequiredPath(
        "wmt22",
        "word_mqm_train_2021_news",
        "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2021-news.tar.gz",
    ),
    RequiredPath(
        "wmt22",
        "word_mqm_train_2021_ted",
        "train-dev_data/task1_word-level/train/en-de_mqm/en-de-train-2021-ted.tar.gz",
    ),
    RequiredPath(
        "wmt22",
        "word_mqm_dev_2022",
        "train-dev_data/task1_word-level/dev/en-de_mqm/en-de-dev-2022.zip",
    ),
]


WMT23_REQUIRED_PATHS = [
    RequiredPath("wmt23", "task2_train_2020", "task_2/train/2020_en-de_processed.tsv"),
    RequiredPath("wmt23", "task2_train_2021_ted", "task_2/train/2021_TED_en-de_processed.tsv"),
    RequiredPath("wmt23", "task2_train_2021_news", "task_2/train/2021_en-de_processed.tsv"),
    RequiredPath("wmt23", "task2_dev_2022", "task_2/dev/2022_en-de_dev_processed.tsv"),
    RequiredPath("wmt23", "task2_test_2022", "task_2/dev/2022_en-de_test_processed.tsv"),
    RequiredPath("wmt23", "hallucination_sentence_gold", "gold_labels/hallucinations_gold_T1s.tsv"),
    RequiredPath("wmt23", "hallucination_word_gold", "gold_labels/hallucinations_gold_T1w.tsv"),
    RequiredPath("wmt23", "hallucination_span_gold", "gold_labels/hallucinations_gold_T2.tsv"),
]


def validate_wmt_roots(
    wmt22_root: str | Path,
    wmt23_root: str | Path,
) -> DataValidationReport:
    wmt22_path = Path(wmt22_root)
    wmt23_path = Path(wmt23_root)
    missing: list[DataValidationIssue] = []

    for required in WMT22_REQUIRED_PATHS:
        absolute = wmt22_path / required.relative_path
        if not absolute.exists():
            missing.append(
                DataValidationIssue(
                    dataset=required.dataset,
                    purpose=required.purpose,
                    path=str(absolute),
                )
            )

    for required in WMT23_REQUIRED_PATHS:
        absolute = wmt23_path / required.relative_path
        if not absolute.exists():
            missing.append(
                DataValidationIssue(
                    dataset=required.dataset,
                    purpose=required.purpose,
                    path=str(absolute),
                )
            )

    return DataValidationReport(
        ok=not missing,
        wmt22_root=str(wmt22_path),
        wmt23_root=str(wmt23_path),
        checked_paths=len(WMT22_REQUIRED_PATHS) + len(WMT23_REQUIRED_PATHS),
        missing=missing,
    )


def raise_for_missing_data(report: DataValidationReport) -> None:
    if report.ok:
        return
    missing_preview = "\n".join(f"- {issue.path}" for issue in report.missing[:10])
    remainder = len(report.missing) - 10
    if remainder > 0:
        missing_preview += f"\n- ... and {remainder} more"
    raise FileNotFoundError(
        "Missing required WMT en-de data files. Initialize the data submodules with "
        "`git submodule update --init --recursive`, then rerun the command.\n"
        f"{missing_preview}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate WMT22/WMT23 en-de raw data roots.")
    parser.add_argument("--wmt22-root", default="data/raw/wmt22")
    parser.add_argument("--wmt23-root", default="data/raw/wmt23")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = validate_wmt_roots(args.wmt22_root, args.wmt23_root)
    print(json.dumps(report.to_dict(), indent=2))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
