from __future__ import annotations

import unittest

from finley.data.session import (
    _extract_session_array,
    _extract_pos_feature_map,
    _normalize_task_type,
    _scalarize,
    _to_python_number,
    _to_python_string,
    build_run_cell_model_rows,
)


class FakeArray:
    def __init__(self, items: list, shape: tuple[int, ...]):
        self._items = items
        self.shape = shape
        self.size = len(items)
        self.flat = items

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row_index, col_index = index
            if isinstance(row_index, slice):
                row_indices = range(*row_index.indices(self.shape[0]))
                values = [self._items[row * self.shape[1] + col_index] for row in row_indices]
                return FakeArray(values, (len(values),))
            width = self.shape[1]
            return self._items[row_index * width + col_index]
        return self._items[index]


class SessionSummaryHelperTests(unittest.TestCase):
    def test_scalarize_unwraps_singleton_arrays(self) -> None:
        wrapped = FakeArray([FakeArray(["run"], (1,))], (1, 1))
        self.assertEqual(_scalarize(wrapped), "run")

    def test_to_python_string_handles_singleton_arrays(self) -> None:
        wrapped = FakeArray(["TrackB"], (1,))
        self.assertEqual(_to_python_string(wrapped), "TrackB")

    def test_extract_session_array_prefers_requested_session(self) -> None:
        target = {"epochs": 7}
        root = FakeArray([[], [], target], (1, 3))
        self.assertIs(_extract_session_array({"task": root}, "task", 3), target)

    def test_to_python_number_handles_singleton_arrays(self) -> None:
        wrapped = FakeArray([7], (1,))
        self.assertEqual(_to_python_number(wrapped), 7)

    def test_normalize_task_type_lowercases_string(self) -> None:
        wrapped = FakeArray(["Run"], (1,))
        self.assertEqual(_normalize_task_type(wrapped), "run")

    def test_build_run_cell_model_rows_filters_and_joins(self) -> None:
        epoch_rows = [
            {
                "session": 3,
                "epoch": 1,
                "task_type": "sleep",
                "task_exposure": "",
                "task_experimentday": "",
                "pos_rows": 10,
                "epoch_duration_sec": None,
                "mean_speed": None,
                "std_speed": None,
                "max_speed": None,
                "speed_q25": None,
                "speed_q50": None,
                "speed_q75": None,
                "moving_fraction": None,
                "fast_fraction": None,
                "path_length": None,
                "step_length_mean": None,
                "step_length_max": None,
                "x_range": None,
                "y_range": None,
                "rawpos_rows": 10,
                "spike_tetrode_count": 1,
                "spike_cell_count": 1,
                "spike_event_rows": 5,
            },
            {
                "session": 3,
                "epoch": 2,
                "task_type": "run",
                "task_exposure": 1,
                "task_experimentday": 7,
                "pos_rows": 20,
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
                "rawpos_rows": 21,
                "spike_tetrode_count": 2,
                "spike_cell_count": 3,
                "spike_event_rows": 12,
            },
        ]
        cell_rows = [
            {
                "animal": "Bon",
                "session": 3,
                "epoch": 1,
                "task_type": "sleep",
                "task_environment": "",
                "tetrode": 1,
                "cell": 1,
                "depth": 100,
                "spikewidth": 8.0,
                "num_spikes": 5,
            },
            {
                "animal": "Bon",
                "session": 3,
                "epoch": 2,
                "task_type": "run",
                "task_environment": "TrackA",
                "tetrode": 2,
                "cell": 1,
                "depth": 110,
                "spikewidth": 9.0,
                "num_spikes": 11,
            },
        ]

        rows = build_run_cell_model_rows(epoch_rows, cell_rows)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_environment"], "TrackA")
        self.assertEqual(rows[0]["task_exposure"], 1)
        self.assertEqual(rows[0]["num_spikes"], 11)
        self.assertEqual(rows[0]["mean_speed"], 1.5)
        self.assertAlmostEqual(rows[0]["firing_rate_hz"], 1.1)

    def test_extract_pos_feature_map_from_mock_pos_struct(self) -> None:
        class PosStruct:
            def __init__(self):
                self.fields = "time x y dir vel smooth-v q-time"
                self.data = FakeArray(
                    [
                        0.0, 1.0, 2.0, 10.0, 0.0, 0.0, 0.0,
                        1.0, 2.0, 4.0, 20.0, 3.0, 3.0, 1.0,
                        2.0, 4.0, 8.0, 30.0, 5.0, 5.0, 2.0,
                    ],
                    (3, 7),
                )

        features = _extract_pos_feature_map(PosStruct())
        self.assertEqual(features["epoch_duration_sec"], 2.0)
        self.assertEqual(features["mean_speed"], 8.0 / 3.0)
        self.assertEqual(features["max_speed"], 5.0)
        self.assertEqual(features["speed_q50"], 3.0)
        self.assertEqual(features["fast_fraction"], 1 / 3)
        self.assertAlmostEqual(features["path_length"], 3 * (5 ** 0.5))
        self.assertEqual(features["x_range"], 3.0)
        self.assertEqual(features["y_range"], 6.0)


if __name__ == "__main__":
    unittest.main()
