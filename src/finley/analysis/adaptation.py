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


__all__ = [
    "get_within_session_unit_key",
    "list_session_epochs",
    "split_session_adaptation_rows",
    "summarize_errors",
    "summarize_unit_overlap",
]
