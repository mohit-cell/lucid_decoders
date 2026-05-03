from __future__ import annotations

import unittest

from lucid_decoders.pipeline import STAGE_ORDER, resolve_stages


class PipelineTests(unittest.TestCase):
    def test_resolve_all_stages(self) -> None:
        self.assertEqual(resolve_stages("all"), STAGE_ORDER)

    def test_resolve_single_stage(self) -> None:
        self.assertEqual(resolve_stages("extract"), ("extract",))


if __name__ == "__main__":
    unittest.main()
