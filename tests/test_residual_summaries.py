from __future__ import annotations

import unittest

from finley.analysis.residuals import summarize_session_residuals, summarize_top_error_cells


class ResidualSummaryTests(unittest.TestCase):
    def test_summarize_top_error_cells_groups_repeated_cells(self) -> None:
        rows = [
            {"session": 6, "tetrode": 13, "cell": 1, "abs_error": 0.8},
            {"session": 6, "tetrode": 13, "cell": 1, "abs_error": 0.6},
            {"session": 6, "tetrode": 18, "cell": 2, "abs_error": 0.7},
        ]
        summary = summarize_top_error_cells(rows)
        self.assertEqual(summary[0]["session"], 6)
        self.assertEqual(summary[0]["tetrode"], 13)
        self.assertEqual(summary[0]["cell"], 1)
        self.assertEqual(summary[0]["row_count"], 2)
        self.assertAlmostEqual(summary[0]["abs_error_sum"], 1.4)

    def test_summarize_session_residuals_computes_mae_rmse(self) -> None:
        rows = [
            {"session": 6, "abs_error": 0.5},
            {"session": 6, "abs_error": 1.5},
            {"session": 7, "abs_error": 1.0},
        ]
        summary = summarize_session_residuals(rows)
        self.assertEqual(summary[0]["session"], 6)
        self.assertAlmostEqual(summary[0]["mae"], 1.0)
        self.assertAlmostEqual(summary[0]["rmse"], (1.25) ** 0.5)
        self.assertEqual(summary[1]["session"], 7)
        self.assertAlmostEqual(summary[1]["mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
