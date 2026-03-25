from __future__ import annotations

import unittest

from finley.analysis.session_profile import summarize_model_table_by_session


class SessionProfileTests(unittest.TestCase):
    def test_summarize_model_table_by_session_builds_profiles_and_groups(self) -> None:
        rows = [
            {
                "session": 3,
                "epoch": 1,
                "task_environment": "TrackA",
                "epoch_duration_sec": 10.0,
                "mean_speed": 1.0,
                "std_speed": 0.2,
                "max_speed": 2.0,
                "mean_accel": 0.1,
                "std_accel": 0.05,
                "mean_abs_accel": 0.1,
                "max_abs_accel": 0.2,
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
                "num_spikes": 2,
                "firing_rate_hz": 0.2,
                "log_firing_rate_hz": 0.18,
            },
            {
                "session": 4,
                "epoch": 1,
                "task_environment": "TrackB",
                "epoch_duration_sec": 12.0,
                "mean_speed": 1.5,
                "std_speed": 0.3,
                "max_speed": 2.5,
                "mean_accel": 0.2,
                "std_accel": 0.07,
                "mean_abs_accel": 0.2,
                "max_abs_accel": 0.4,
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
                "num_spikes": 3,
                "firing_rate_hz": 0.25,
                "log_firing_rate_hz": 0.22,
            },
        ]
        summary = summarize_model_table_by_session(rows, hard_sessions=[4], easy_sessions=[3])
        self.assertEqual(len(summary["session_profiles"]), 2)
        self.assertEqual(len(summary["session_track_profiles"]), 2)
        self.assertIn("group_summaries", summary)
        self.assertIn("group_comparison", summary)
        self.assertEqual(summary["session_profiles"][0]["session"], 3)
        self.assertAlmostEqual(summary["session_profiles"][0]["mean_speed_mean"], 1.0)
        self.assertEqual(summary["session_track_profiles"][0]["task_environment"], "TrackA")
        self.assertAlmostEqual(
            summary["group_comparison"]["hard_minus_easy"]["mean_speed_mean"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
