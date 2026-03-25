from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.residuals import summarize_session_residuals, summarize_top_error_cells
from finley.models.run_cell_baseline import load_model_table, split_by_session
from finley.models.run_cell_nonlinear import (
    TreeRegressorConfig,
    build_feature_matrix,
    filter_rows_for_target,
    predict_forest,
    fit_random_forest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect leave-one-session-out residuals for hard sessions with the nonlinear model."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hard_session_residuals_nonlinear.json",
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
        "--feature-groups",
        nargs="+",
        default=["movement_summaries", "population_context", "cell_metadata"],
        help="Feature groups to include.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top residual rows/cells to keep per session.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=48,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of each tree.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=8,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Feature subsampling per split: sqrt, all, or a positive integer.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for deterministic fitting.",
    )
    return parser.parse_args()


def parse_max_features(value: str) -> str | int:
    if value in {"sqrt", "all"}:
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--max-features must be sqrt, all, or a positive integer.")
    return parsed


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
    feature_groups: list[str],
    config: TreeRegressorConfig,
) -> list[dict]:
    split = split_by_session(rows, held_out_session=held_out_session)
    train_rows, _ = filter_rows_for_target(split.train_rows, target_column)
    test_rows, _ = filter_rows_for_target(split.test_rows, target_column)
    x_train = build_feature_matrix(train_rows, feature_groups=feature_groups)
    y_train = [float(row[target_column]) for row in train_rows]
    x_test = build_feature_matrix(test_rows, feature_groups=feature_groups)

    forest = fit_random_forest(x_train, y_train, config=config)
    predictions = predict_forest(x_test, forest)

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
    config = TreeRegressorConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        random_seed=args.random_seed,
    )

    all_residual_rows: list[dict] = []
    top_rows: list[dict] = []
    top_cells: list[dict] = []
    for session in args.sessions:
        session_rows = _predict_rows(
            rows,
            held_out_session=session,
            target_column=args.target,
            feature_groups=args.feature_groups,
            config=config,
        )
        all_residual_rows.extend(session_rows)
        top_rows.extend(session_rows[: args.top_n])
        top_cells.extend(summarize_top_error_cells(session_rows)[: args.top_n])

    payload = {
        "target_column": args.target,
        "feature_groups": args.feature_groups,
        "sessions": args.sessions,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "max_features": config.max_features,
        "random_seed": config.random_seed,
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
    print(f"Wrote nonlinear hard-session residuals to {output_path}")
    print(f"Wrote top residual row table to {rows_csv}")
    print(f"Wrote top error cell table to {cells_csv}")
    print(f"Wrote session metrics table to {metrics_csv}")


if __name__ == "__main__":
    main()
