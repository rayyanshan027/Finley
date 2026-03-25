from __future__ import annotations

import unittest

from finley.data.session import (
    _extract_session_array,
    _normalize_task_type,
    _scalarize,
    _to_python_number,
    _to_python_string,
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


if __name__ == "__main__":
    unittest.main()
