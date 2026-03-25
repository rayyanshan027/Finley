from __future__ import annotations

import unittest

from finley.analysis.adaptation import (
    apply_unit_residual_offsets,
    fit_unit_residual_offsets,
    list_session_epochs,
    split_session_adaptation_rows,
    summarize_unit_overlap,
)
from scripts.run_session_adaptation_experiment import resolve_unit_residual_shrinkages


class AdaptationTests(unittest.TestCase):
    def test_list_session_epochs_returns_sorted_unique_epochs(self) -> None:
        rows = [
            {"session": 6, "epoch": 4},
            {"session": 6, "epoch": 2},
            {"session": 6, "epoch": 4},
            {"session": 7, "epoch": 1},
        ]
        self.assertEqual(list_session_epochs(rows, 6), [2, 4])

    def test_split_session_adaptation_rows_uses_earliest_epochs_for_adaptation(self) -> None:
        rows = [
            {"session": 5, "epoch": 1, "value": "train"},
            {"session": 6, "epoch": 2, "value": "a"},
            {"session": 6, "epoch": 4, "value": "b"},
            {"session": 6, "epoch": 6, "value": "c"},
        ]
        train_rows, test_rows, adaptation_epochs, evaluation_epochs = split_session_adaptation_rows(
            rows,
            held_out_session=6,
            adaptation_epoch_count=1,
        )
        self.assertEqual(adaptation_epochs, [2])
        self.assertEqual(evaluation_epochs, [4, 6])
        self.assertEqual(len(train_rows), 2)
        self.assertEqual(len(test_rows), 2)

    def test_split_session_adaptation_rows_rejects_full_adaptation(self) -> None:
        rows = [
            {"session": 6, "epoch": 2},
            {"session": 6, "epoch": 4},
        ]
        with self.assertRaises(ValueError):
            split_session_adaptation_rows(rows, held_out_session=6, adaptation_epoch_count=2)

    def test_summarize_unit_overlap_reports_unit_and_row_coverage(self) -> None:
        adaptation_rows = [
            {"session": 6, "epoch": 2, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 2, "tetrode": 1, "cell": 2},
            {"session": 6, "epoch": 4, "tetrode": 1, "cell": 1},
        ]
        evaluation_rows = [
            {"session": 6, "epoch": 6, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 6, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 6, "tetrode": 1, "cell": 3},
        ]
        summary = summarize_unit_overlap(adaptation_rows, evaluation_rows)
        self.assertEqual(summary["adaptation_unit_count"], 2)
        self.assertEqual(summary["evaluation_unit_count"], 2)
        self.assertEqual(summary["shared_unit_count"], 1)
        self.assertAlmostEqual(summary["shared_unit_fraction"], 0.5)
        self.assertEqual(summary["evaluation_rows_on_seen_units"], 2)
        self.assertAlmostEqual(summary["evaluation_row_seen_unit_fraction"], 2 / 3)

    def test_fit_unit_residual_offsets_applies_shrinkage_per_unit(self) -> None:
        rows = [
            {"session": 6, "epoch": 2, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 4, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 2, "tetrode": 1, "cell": 2},
        ]
        residuals = [0.6, 0.2, -0.3]
        offsets = fit_unit_residual_offsets(rows, residuals, shrinkage=2.0)
        self.assertAlmostEqual(offsets[(1, 1)], 0.2)
        self.assertAlmostEqual(offsets[(1, 2)], -0.1)

    def test_apply_unit_residual_offsets_only_changes_seen_units(self) -> None:
        rows = [
            {"session": 6, "epoch": 6, "tetrode": 1, "cell": 1},
            {"session": 6, "epoch": 6, "tetrode": 1, "cell": 3},
        ]
        predictions, corrected_count = apply_unit_residual_offsets(
            rows,
            [1.0, 2.0],
            {(1, 1): 0.25},
        )
        self.assertEqual(predictions, [1.25, 2.0])
        self.assertEqual(corrected_count, 1)

    def test_resolve_unit_residual_shrinkages_only_sweeps_residual_variant(self) -> None:
        self.assertEqual(
            resolve_unit_residual_shrinkages("baseline_plus_unit_residual", [0.0, 4.0, 8.0]),
            [0.0, 4.0, 8.0],
        )
        self.assertEqual(
            resolve_unit_residual_shrinkages("baseline", [0.0, 4.0, 8.0]),
            [0.0],
        )


if __name__ == "__main__":
    unittest.main()
