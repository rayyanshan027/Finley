from __future__ import annotations

import unittest

from finley.analysis.hard_sessions import (
    build_epoch_rows,
    build_hard_session_diagnostics,
)


def make_row(
    *,
    session: int,
    epoch: int,
    environment: str,
    cell: int,
    target: float,
    mean_speed: float,
    mean_abs_accel: float,
    spike_cell_count: int,
    spike_event_rows_epoch: int,
) -> dict:
    return {
        "animal": "Bon",
        "session": session,
        "epoch": epoch,
        "task_environment": environment,
        "task_exposure": 1,
        "task_experimentday": 7,
        "pos_rows": 20,
        "epoch_duration_sec": 10.0,
        "mean_speed": mean_speed,
        "std_speed": 0.2,
        "max_speed": mean_speed + 1.0,
        "mean_accel": 0.1,
        "std_accel": 0.05,
        "mean_abs_accel": mean_abs_accel,
        "max_abs_accel": mean_abs_accel + 0.2,
        "speed_q25": mean_speed - 0.5,
        "speed_q50": mean_speed,
        "speed_q75": mean_speed + 0.5,
        "moving_fraction": 0.6,
        "fast_fraction": 0.2,
        "path_length": 30.0 + mean_speed,
        "step_length_mean": 2.0,
        "step_length_max": 3.0,
        "x_range": 8.0,
        "y_range": 4.0,
        "rawpos_rows": 22,
        "spike_tetrode_count": 2,
        "spike_cell_count": spike_cell_count,
        "spike_event_rows_epoch": spike_event_rows_epoch,
        "tetrode": 1,
        "cell": cell,
        "depth": 100.0,
        "spikewidth": 8.0,
        "num_spikes": int(target * 10),
        "firing_rate_hz": target,
        "log_num_spikes": target,
        "log_firing_rate_hz": target,
        "session_centered_log_firing_rate_hz": 0.0,
    }


class HardSessionDiagnosticsTests(unittest.TestCase):
    def test_build_epoch_rows_deduplicates_epochs(self) -> None:
        rows = [
            make_row(
                session=6,
                epoch=1,
                environment="TrackB",
                cell=1,
                target=1.5,
                mean_speed=2.0,
                mean_abs_accel=0.8,
                spike_cell_count=5,
                spike_event_rows_epoch=40,
            ),
            make_row(
                session=6,
                epoch=1,
                environment="TrackB",
                cell=2,
                target=1.7,
                mean_speed=2.0,
                mean_abs_accel=0.8,
                spike_cell_count=5,
                spike_event_rows_epoch=40,
            ),
        ]
        epoch_rows = build_epoch_rows(rows)
        self.assertEqual(len(epoch_rows), 1)
        self.assertEqual(epoch_rows[0]["epoch"], 1)

    def test_build_hard_session_diagnostics_returns_group_deltas_and_outliers(self) -> None:
        rows = [
            make_row(
                session=3,
                epoch=1,
                environment="TrackA",
                cell=1,
                target=0.6,
                mean_speed=1.0,
                mean_abs_accel=0.2,
                spike_cell_count=2,
                spike_event_rows_epoch=12,
            ),
            make_row(
                session=4,
                epoch=1,
                environment="TrackA",
                cell=1,
                target=0.7,
                mean_speed=1.1,
                mean_abs_accel=0.25,
                spike_cell_count=2,
                spike_event_rows_epoch=13,
            ),
            make_row(
                session=6,
                epoch=2,
                environment="TrackB",
                cell=1,
                target=1.6,
                mean_speed=2.4,
                mean_abs_accel=0.9,
                spike_cell_count=6,
                spike_event_rows_epoch=45,
            ),
            make_row(
                session=7,
                epoch=3,
                environment="TrackB",
                cell=1,
                target=1.7,
                mean_speed=2.5,
                mean_abs_accel=1.0,
                spike_cell_count=6,
                spike_event_rows_epoch=47,
            ),
            make_row(
                session=9,
                epoch=4,
                environment="TrackB",
                cell=1,
                target=2.1,
                mean_speed=3.1,
                mean_abs_accel=1.4,
                spike_cell_count=8,
                spike_event_rows_epoch=60,
            ),
        ]

        diagnostics = build_hard_session_diagnostics(
            rows,
            hard_sessions=[6, 7, 9],
            easy_sessions=[3, 4],
            top_n=2,
        )

        self.assertEqual(diagnostics["row_count"], 5)
        self.assertEqual(diagnostics["epoch_row_count"], 5)
        self.assertIn("session_summaries", diagnostics)
        self.assertIn("group_summaries", diagnostics)
        self.assertIn("group_deltas", diagnostics)
        self.assertIn("outlier_rows", diagnostics)
        self.assertTrue(diagnostics["outlier_rows"])
        row_deltas = diagnostics["group_deltas"]["row_metrics_hard_minus_easy"]
        self.assertGreater(row_deltas["mean_speed_mean"], 0.0)
        self.assertGreater(row_deltas["log_firing_rate_hz_q90"], 0.0)


if __name__ == "__main__":
    unittest.main()
