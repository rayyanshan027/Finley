from __future__ import annotations

import math


def get_within_session_unit_key(row: dict) -> tuple[int, int]:
    return int(row["tetrode"]), int(row["cell"])


def list_session_epochs(rows: list[dict], held_out_session: int) -> list[int]:
    return sorted({int(row["epoch"]) for row in rows if int(row["session"]) == held_out_session})


def split_session_adaptation_rows(
    rows: list[dict],
    held_out_session: int,
    adaptation_epoch_count: int,
) -> tuple[list[dict], list[dict], list[int], list[int]]:
    if adaptation_epoch_count < 0:
        raise ValueError("adaptation_epoch_count must be non-negative.")

    held_out_epochs = list_session_epochs(rows, held_out_session)
    if not held_out_epochs:
        raise ValueError(f"No rows found for held_out_session={held_out_session}.")
    if adaptation_epoch_count >= len(held_out_epochs):
        raise ValueError("adaptation_epoch_count must be smaller than the number of held-out epochs.")

    adaptation_epochs = held_out_epochs[:adaptation_epoch_count]
    evaluation_epochs = held_out_epochs[adaptation_epoch_count:]
    train_rows = [
        row
        for row in rows
        if int(row["session"]) != held_out_session or int(row["epoch"]) in set(adaptation_epochs)
    ]
    test_rows = [
        row
        for row in rows
        if int(row["session"]) == held_out_session and int(row["epoch"]) in set(evaluation_epochs)
    ]
    if not test_rows:
        raise ValueError("No evaluation rows remain after adaptation split.")
    return train_rows, test_rows, adaptation_epochs, evaluation_epochs


def summarize_errors(errors: list[float]) -> dict[str, float]:
    abs_errors = [abs(error) for error in errors]
    sq_errors = [error * error for error in errors]
    return {
        "mae": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
        "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0,
    }


def summarize_unit_overlap(adaptation_rows: list[dict], evaluation_rows: list[dict]) -> dict[str, float | int]:
    adaptation_units = {get_within_session_unit_key(row) for row in adaptation_rows}
    evaluation_units = {get_within_session_unit_key(row) for row in evaluation_rows}
    shared_units = adaptation_units & evaluation_units
    evaluation_rows_on_seen_units = sum(
        1 for row in evaluation_rows if get_within_session_unit_key(row) in adaptation_units
    )
    return {
        "adaptation_unit_count": len(adaptation_units),
        "evaluation_unit_count": len(evaluation_units),
        "shared_unit_count": len(shared_units),
        "shared_unit_fraction": (
            len(shared_units) / len(evaluation_units) if evaluation_units else 0.0
        ),
        "evaluation_rows_on_seen_units": evaluation_rows_on_seen_units,
        "evaluation_row_seen_unit_fraction": (
            evaluation_rows_on_seen_units / len(evaluation_rows) if evaluation_rows else 0.0
        ),
    }


def fit_unit_residual_offsets(
    rows: list[dict],
    residuals: list[float],
    shrinkage: float = 4.0,
) -> dict[tuple[int, int], float]:
    if shrinkage < 0:
        raise ValueError("shrinkage must be non-negative.")
    if len(rows) != len(residuals):
        raise ValueError("rows and residuals must have the same length.")

    residual_sums: dict[tuple[int, int], float] = {}
    residual_counts: dict[tuple[int, int], int] = {}
    for row, residual in zip(rows, residuals):
        key = get_within_session_unit_key(row)
        residual_sums[key] = residual_sums.get(key, 0.0) + float(residual)
        residual_counts[key] = residual_counts.get(key, 0) + 1

    offsets: dict[tuple[int, int], float] = {}
    for key, residual_sum in residual_sums.items():
        count = residual_counts[key]
        mean_residual = residual_sum / count
        offsets[key] = (count / (count + shrinkage)) * mean_residual
    return offsets


def apply_unit_residual_offsets(
    rows: list[dict],
    predictions: list[float],
    offsets: dict[tuple[int, int], float],
) -> tuple[list[float], int]:
    if len(rows) != len(predictions):
        raise ValueError("rows and predictions must have the same length.")

    corrected_predictions: list[float] = []
    corrected_row_count = 0
    for row, prediction in zip(rows, predictions):
        offset = offsets.get(get_within_session_unit_key(row), 0.0)
        if offset != 0.0:
            corrected_row_count += 1
        corrected_predictions.append(float(prediction) + offset)
    return corrected_predictions, corrected_row_count


__all__ = [
    "apply_unit_residual_offsets",
    "fit_unit_residual_offsets",
    "get_within_session_unit_key",
    "list_session_epochs",
    "split_session_adaptation_rows",
    "summarize_errors",
    "summarize_unit_overlap",
]
