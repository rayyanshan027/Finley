from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.residuals import summarize_session_residuals, summarize_top_error_cells
from finley.models.run_cell_baseline import (
    apply_feature_scaler,
    build_design_matrix,
    filter_rows_for_target,
    fit_feature_scaler,
    fit_ridge_regression,
    load_model_table,
    predict,
    split_by_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect leave-one-session-out residuals for hard sessions."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hard_session_residuals.json",
        help="Path to JSON output.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[6, 7, 9],
        help="Held-out sessions to inspect.",
    )
    parser.add_argument(
        "--target",
        default="log_firing_rate_hz",
        choices=[
            "log_num_spikes",
            "num_spikes",
            "firing_rate_hz",
            "log_firing_rate_hz",
            "session_centered_log_firing_rate_hz",
        ],
        help="Target column to predict.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=100.0,
        help="Ridge regularization strength.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["movement_summaries"],
        help="Feature groups to include.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top residual rows/cells to keep per session.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _predict_rows(
    rows: list[dict],
    held_out_session: int,
    target_column: str,
    ridge_alpha: float,
    feature_groups: list[str],
) -> list[dict]:
    split = split_by_session(rows, held_out_session=held_out_session)
    train_rows, _ = filter_rows_for_target(split.train_rows, target_column)
    test_rows, _ = filter_rows_for_target(split.test_rows, target_column)
    x_train, y_train = build_design_matrix(
        train_rows,
        target_column=target_column,
        feature_groups=feature_groups,
    )
    x_test, _ = build_design_matrix(
        test_rows,
        target_column=target_column,
        feature_groups=feature_groups,
    )
    scaler = fit_feature_scaler(x_train)
    coefficients = fit_ridge_regression(
        apply_feature_scaler(x_train, scaler),
        y_train,
        alpha=ridge_alpha,
    )
    predictions = predict(apply_feature_scaler(x_test, scaler), coefficients)

    output: list[dict] = []
    for row, prediction in zip(test_rows, predictions):
        actual = float(row[target_column])
        error = float(prediction) - actual
        output.append(
            {
                "session": int(row["session"]),
                "epoch": int(row["epoch"]),
                "task_environment": row["task_environment"],
                "tetrode": int(row["tetrode"]),
                "cell": int(row["cell"]),
                "target_column": target_column,
                "actual": actual,
                "prediction": float(prediction),
                "error": error,
                "abs_error": abs(error),
                "num_spikes": int(row["num_spikes"]),
                "firing_rate_hz": row.get("firing_rate_hz"),
                "log_firing_rate_hz": row.get("log_firing_rate_hz"),
                "mean_speed": row.get("mean_speed"),
                "mean_abs_accel": row.get("mean_abs_accel"),
                "spike_cell_count": int(row["spike_cell_count"]),
                "spike_event_rows_epoch": int(row["spike_event_rows_epoch"]),
            }
        )
    output.sort(key=lambda row: float(row["abs_error"]), reverse=True)
    return output


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)

    all_residual_rows: list[dict] = []
    top_rows: list[dict] = []
    top_cells: list[dict] = []
    for session in args.sessions:
        session_rows = _predict_rows(
            rows,
            held_out_session=session,
            target_column=args.target,
            ridge_alpha=args.ridge_alpha,
            feature_groups=args.feature_groups,
        )
        all_residual_rows.extend(session_rows)

        for row in session_rows[: args.top_n]:
            top_rows.append(row)

        for row in summarize_top_error_cells(session_rows)[: args.top_n]:
            top_cells.append(row)

    payload = {
        "target_column": args.target,
        "ridge_alpha": args.ridge_alpha,
        "feature_groups": args.feature_groups,
        "sessions": args.sessions,
        "session_metrics": summarize_session_residuals(all_residual_rows),
        "top_residual_rows": top_rows,
        "top_error_cells": top_cells,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows_csv = output_path.with_suffix(".rows.csv")
    cells_csv = output_path.with_suffix(".cells.csv")
    metrics_csv = output_path.with_suffix(".sessions.csv")
    write_csv(rows_csv, top_rows)
    write_csv(cells_csv, top_cells)
    write_csv(metrics_csv, payload["session_metrics"])

    print(json.dumps(payload, indent=2))
    print(f"Wrote hard-session residuals to {output_path}")
    print(f"Wrote top residual row table to {rows_csv}")
    print(f"Wrote top error cell table to {cells_csv}")
    print(f"Wrote session metrics table to {metrics_csv}")


if __name__ == "__main__":
    main()
