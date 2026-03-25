from __future__ import annotations

import unittest

from finley.models.run_cell_baseline import build_design_matrix, split_by_session


class RunCellBaselineTests(unittest.TestCase):
    def test_split_by_session_holds_out_latest_by_default(self) -> None:
        rows = [
            {"session": 3, "task_environment": "TrackA", "task_exposure": 1, "task_experimentday": 7, "pos_rows": 10, "rawpos_rows": 10, "spike_tetrode_count": 1, "spike_cell_count": 2, "spike_event_rows_epoch": 5, "tetrode": 1, "depth": 100, "spikewidth": 8.0, "num_spikes": 2, "log_num_spikes": 1.0},
            {"session": 4, "task_environment": "TrackB", "task_exposure": 2, "task_experimentday": 8, "pos_rows": 12, "rawpos_rows": 12, "spike_tetrode_count": 2, "spike_cell_count": 3, "spike_event_rows_epoch": 9, "tetrode": 2, "depth": 110, "spikewidth": 9.0, "num_spikes": 3, "log_num_spikes": 1.3},
        ]
        split = split_by_session(rows)
        self.assertEqual(split.held_out_session, 4)
        self.assertEqual(len(split.train_rows), 1)
        self.assertEqual(len(split.test_rows), 1)

    def test_build_design_matrix_shapes(self) -> None:
        rows = [
            {
                "session": 3,
                "task_environment": "TrackA",
                "task_exposure": 1,
                "task_experimentday": 7,
                "pos_rows": 10,
                "rawpos_rows": 10,
                "spike_tetrode_count": 1,
                "spike_cell_count": 2,
                "spike_event_rows_epoch": 5,
                "tetrode": 1,
                "depth": 100,
                "spikewidth": 8.0,
                "num_spikes": 2,
                "log_num_spikes": 1.0,
            }
        ]
        x, y = build_design_matrix(rows, target_column="log_num_spikes")
        self.assertEqual(len(x), 1)
        self.assertEqual(len(x[0]), 13)
        self.assertEqual(len(y), 1)


if __name__ == "__main__":
    unittest.main()
