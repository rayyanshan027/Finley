from __future__ import annotations

import math


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


__all__ = ["list_session_epochs", "split_session_adaptation_rows", "summarize_errors"]
