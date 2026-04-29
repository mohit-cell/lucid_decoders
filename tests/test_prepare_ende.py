from __future__ import annotations

import unittest

from lucid_decoders.data.prepare_ende import merge_spans, parse_task2_span_fields, tokens_to_text_and_spans
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


if __name__ == "__main__":
    unittest.main()
