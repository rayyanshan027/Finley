from __future__ import annotations

import unittest

from finley.data.session import (
    _extract_session_array,
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
        row, col = index
        width = self.shape[1]
        return self._items[row * width + col]


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


if __name__ == "__main__":
    unittest.main()
