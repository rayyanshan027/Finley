from __future__ import annotations

import unittest

from finley.models.run_cell_baseline import (
    apply_feature_scaler,
    build_design_matrix,
    compute_metrics,
    filter_rows_for_target,
    fit_feature_scaler,
    split_by_session,
)


class RunCellBaselineTests(unittest.TestCase):
    def test_split_by_session_holds_out_latest_by_default(self) -> None:
        rows = [
            {"session": 3, "task_environment": "TrackA", "task_exposure": 1, "task_experimentday": 7, "pos_rows": 10, "epoch_duration_sec": 9.0, "mean_speed": 1.0, "std_speed": 0.2, "max_speed": 2.0, "speed_q25": 0.5, "speed_q50": 1.0, "speed_q75": 1.5, "moving_fraction": 0.5, "fast_fraction": 0.1, "path_length": 20.0, "step_length_mean": 2.0, "step_length_max": 3.0, "x_range": 10.0, "y_range": 5.0, "rawpos_rows": 10, "spike_tetrode_count": 1, "spike_cell_count": 2, "spike_event_rows_epoch": 5, "tetrode": 1, "depth": 100, "spikewidth": 8.0, "num_spikes": 2, "log_num_spikes": 1.0},
            {"session": 4, "task_environment": "TrackB", "task_exposure": 2, "task_experimentday": 8, "pos_rows": 12, "epoch_duration_sec": 10.0, "mean_speed": 1.5, "std_speed": 0.3, "max_speed": 2.5, "speed_q25": 1.0, "speed_q50": 1.5, "speed_q75": 2.0, "moving_fraction": 0.6, "fast_fraction": 0.2, "path_length": 22.0, "step_length_mean": 2.2, "step_length_max": 3.2, "x_range": 12.0, "y_range": 6.0, "rawpos_rows": 12, "spike_tetrode_count": 2, "spike_cell_count": 3, "spike_event_rows_epoch": 9, "tetrode": 2, "depth": 110, "spikewidth": 9.0, "num_spikes": 3, "log_num_spikes": 1.3},
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
                "epoch_duration_sec": 9.0,
                "mean_speed": 1.0,
                "std_speed": 0.2,
                "max_speed": 2.0,
                "speed_q25": 0.5,
                "speed_q50": 1.0,
                "speed_q75": 1.5,
                "moving_fraction": 0.5,
                "fast_fraction": 0.1,
                "path_length": 20.0,
                "step_length_mean": 2.0,
                "step_length_max": 3.0,
                "x_range": 10.0,
                "y_range": 5.0,
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
        self.assertEqual(len(x[0]), 27)
        self.assertEqual(len(y), 1)

    def test_build_design_matrix_rejects_missing_target_values(self) -> None:
        rows = [
            {
                "session": 3,
                "task_environment": "TrackA",
                "task_exposure": 1,
                "task_experimentday": 7,
                "pos_rows": 10,
                "epoch_duration_sec": 9.0,
                "mean_speed": 1.0,
                "std_speed": 0.2,
                "max_speed": 2.0,
                "speed_q25": 0.5,
                "speed_q50": 1.0,
                "speed_q75": 1.5,
                "moving_fraction": 0.5,
                "fast_fraction": 0.1,
                "path_length": 20.0,
                "step_length_mean": 2.0,
                "step_length_max": 3.0,
                "x_range": 10.0,
                "y_range": 5.0,
                "rawpos_rows": 10,
                "spike_tetrode_count": 1,
                "spike_cell_count": 2,
                "spike_event_rows_epoch": 5,
                "tetrode": 1,
                "depth": 100,
                "spikewidth": 8.0,
                "num_spikes": 2,
                "log_firing_rate_hz": None,
            }
        ]
        with self.assertRaises(ValueError):
            build_design_matrix(rows, target_column="log_firing_rate_hz")

    def test_filter_rows_for_target_drops_missing_values(self) -> None:
        rows = [{"log_firing_rate_hz": 1.0}, {"log_firing_rate_hz": None}]
        filtered_rows, dropped_count = filter_rows_for_target(rows, "log_firing_rate_hz")
        self.assertEqual(len(filtered_rows), 1)
        self.assertEqual(dropped_count, 1)

    def test_compute_metrics_filters_missing_target_rows(self) -> None:
        train_rows = [
            {
                "session": 3,
                "task_environment": "TrackA",
                "task_exposure": 1,
                "task_experimentday": 7,
                "pos_rows": 10,
                "epoch_duration_sec": 9.0,
                "mean_speed": 1.0,
                "std_speed": 0.2,
                "max_speed": 2.0,
                "speed_q25": 0.5,
                "speed_q50": 1.0,
                "speed_q75": 1.5,
                "moving_fraction": 0.5,
                "fast_fraction": 0.1,
                "path_length": 20.0,
                "step_length_mean": 2.0,
                "step_length_max": 3.0,
                "x_range": 10.0,
                "y_range": 5.0,
                "rawpos_rows": 10,
                "spike_tetrode_count": 1,
                "spike_cell_count": 2,
                "spike_event_rows_epoch": 5,
                "tetrode": 1,
                "depth": 100,
                "spikewidth": 8.0,
                "num_spikes": 2,
                "log_firing_rate_hz": 1.0,
            },
            {
                "session": 3,
                "task_environment": "TrackB",
                "task_exposure": 1,
                "task_experimentday": 7,
                "pos_rows": 10,
                "epoch_duration_sec": None,
                "mean_speed": 1.2,
                "std_speed": 0.2,
                "max_speed": 2.2,
                "speed_q25": 0.6,
                "speed_q50": 1.1,
                "speed_q75": 1.6,
                "moving_fraction": 0.4,
                "fast_fraction": 0.1,
                "path_length": 18.0,
                "step_length_mean": 1.8,
                "step_length_max": 2.8,
                "x_range": 9.0,
                "y_range": 4.0,
                "rawpos_rows": 9,
                "spike_tetrode_count": 1,
                "spike_cell_count": 2,
                "spike_event_rows_epoch": 4,
                "tetrode": 1,
                "depth": 90,
                "spikewidth": 7.5,
                "num_spikes": 1,
                "log_firing_rate_hz": None,
            },
        ]
        test_rows = [
            {
                "session": 4,
                "task_environment": "TrackA",
                "task_exposure": 2,
                "task_experimentday": 8,
                "pos_rows": 12,
                "epoch_duration_sec": 10.0,
                "mean_speed": 1.5,
                "std_speed": 0.3,
                "max_speed": 2.5,
                "speed_q25": 1.0,
                "speed_q50": 1.5,
                "speed_q75": 2.0,
                "moving_fraction": 0.6,
                "fast_fraction": 0.2,
                "path_length": 22.0,
                "step_length_mean": 2.2,
                "step_length_max": 3.2,
                "x_range": 12.0,
                "y_range": 6.0,
                "rawpos_rows": 12,
                "spike_tetrode_count": 2,
                "spike_cell_count": 3,
                "spike_event_rows_epoch": 9,
                "tetrode": 2,
                "depth": 110,
                "spikewidth": 9.0,
                "num_spikes": 3,
                "log_firing_rate_hz": 1.2,
            }
        ]
        metrics = compute_metrics(
            train_rows,
            test_rows,
            held_out_session=4,
            target_column="log_firing_rate_hz",
        )
        self.assertEqual(metrics.train_count, 1)
        self.assertEqual(metrics.test_count, 1)
        self.assertEqual(metrics.dropped_train_count, 1)
        self.assertEqual(metrics.dropped_test_count, 0)

    def test_feature_scaler_preserves_intercept_and_scales_columns(self) -> None:
        x = [
            [1.0, 10.0, 100.0],
            [1.0, 20.0, 300.0],
        ]
        scaler = fit_feature_scaler(x)
        transformed = apply_feature_scaler(x, scaler)
        self.assertEqual(transformed[0][0], 1.0)
        self.assertEqual(transformed[1][0], 1.0)
        self.assertAlmostEqual(sum(row[1] for row in transformed) / 2, 0.0)


if __name__ == "__main__":
    unittest.main()
