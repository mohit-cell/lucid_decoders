from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from lucid_decoders.data.prepare_ende import (
    build_wmt23_task2_examples,
    merge_spans,
    parse_task2_span_fields,
    tokens_to_text_and_spans,
)
from lucid_decoders.data.validate_ende import validate_wmt_roots
from lucid_decoders.schemas import HallucinationSpan


class PrepareEnDeTests(unittest.TestCase):
    def test_tokens_to_text_and_spans(self) -> None:
        text, spans = tokens_to_text_and_spans(["Hallo", ",", "Welt"])
        self.assertEqual(text, "Hallo , Welt")
        self.assertEqual(spans, [(0, 5), (6, 7), (8, 12)])

    def test_parse_task2_span_fields_inclusive_end(self) -> None:
        spans, types = parse_task2_span_fields("4 10", "6 11", "major minor")
        self.assertEqual([(span.start, span.end) for span in spans], [(4, 7), (10, 12)])
        self.assertEqual(types, ["major", "minor"])

    def test_merge_spans(self) -> None:
        spans = [
            HallucinationSpan(0, 3),
            HallucinationSpan(2, 5),
            HallucinationSpan(8, 10),
        ]
        merged = merge_spans(spans)
        self.assertEqual([(span.start, span.end) for span in merged], [(0, 5), (8, 10)])

    def test_validate_wmt_roots_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = validate_wmt_roots(root / "wmt22", root / "wmt23")
            self.assertFalse(report.ok)
            self.assertGreater(report.checked_paths, 0)
            self.assertEqual(len(report.missing), report.checked_paths)

    def test_wmt23_task2_parser_handles_quotes_and_trailing_empty_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task_2" / "dev" / "2022_en-de_dev_processed.tsv"
            task_path.parent.mkdir(parents=True)
            task_path.write_text(
                "mt_system\tdoc_id\tseg_id\tannotator_id\tsource\ttarget\tstart_ids\tend_ids\terrors\n"
                "sys\tdoc\t1\trater\t\"quoted source\"\t\"quoted target\"\t-1\t-1\tno-error\t\n",
                encoding="utf-8",
            )

            examples = build_wmt23_task2_examples(root)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].source_text, '"quoted source"')
            self.assertEqual(examples[0].hypothesis_text, '"quoted target"')
            self.assertEqual(examples[0].sentence_label, 0)


if __name__ == "__main__":
    unittest.main()
