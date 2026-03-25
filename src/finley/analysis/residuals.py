from __future__ import annotations

import math


def summarize_top_error_cells(
    rows: list[dict],
    session_column: str = "session",
    tetrode_column: str = "tetrode",
    cell_column: str = "cell",
    abs_error_column: str = "abs_error",
) -> list[dict[str, float | int]]:
    grouped: dict[tuple[int, int, int], dict[str, float | int]] = {}
    for row in rows:
        key = (
            int(row[session_column]),
            int(row[tetrode_column]),
            int(row[cell_column]),
        )
        bucket = grouped.setdefault(
            key,
            {
                "session": key[0],
                "tetrode": key[1],
                "cell": key[2],
                "row_count": 0,
                "abs_error_sum": 0.0,
                "abs_error_max": 0.0,
            },
        )
        abs_error = float(row[abs_error_column])
        bucket["row_count"] = int(bucket["row_count"]) + 1
        bucket["abs_error_sum"] = float(bucket["abs_error_sum"]) + abs_error
        bucket["abs_error_max"] = max(float(bucket["abs_error_max"]), abs_error)

    output: list[dict[str, float | int]] = []
    for bucket in grouped.values():
        row_count = int(bucket["row_count"])
        output.append(
            {
                **bucket,
                "abs_error_mean": float(bucket["abs_error_sum"]) / row_count if row_count else 0.0,
            }
        )
    output.sort(
        key=lambda row: (
            float(row["abs_error_sum"]),
            float(row["abs_error_max"]),
        ),
        reverse=True,
    )
    return output


def summarize_session_residuals(rows: list[dict]) -> list[dict[str, float | int]]:
    sessions = sorted({int(row["session"]) for row in rows})
    output: list[dict[str, float | int]] = []
    for session in sessions:
        session_rows = [row for row in rows if int(row["session"]) == session]
        abs_errors = [float(row["abs_error"]) for row in session_rows]
        sq_errors = [error * error for error in abs_errors]
        output.append(
            {
                "session": session,
                "row_count": len(session_rows),
                "mae": sum(abs_errors) / len(abs_errors) if abs_errors else 0.0,
                "rmse": math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0.0,
            }
        )
    return output


__all__ = ["summarize_session_residuals", "summarize_top_error_cells"]
