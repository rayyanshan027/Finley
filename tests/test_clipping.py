from __future__ import annotations

import unittest

from finley.analysis.clipping import clip_rows_for_target, quantile


class ClippingTests(unittest.TestCase):
    def test_quantile_interpolates(self) -> None:
        self.assertEqual(quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_clip_rows_for_target_caps_only_upper_tail(self) -> None:
        rows = [{"target": 1.0}, {"target": 2.0}, {"target": 100.0}]
        clipped_rows, clip_value = clip_rows_for_target(rows, "target", 0.5)
        self.assertEqual(clip_value, 2.0)
        self.assertEqual([row["target"] for row in clipped_rows], [1.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
