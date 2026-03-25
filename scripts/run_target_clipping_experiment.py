from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.clipping import clip_rows_for_target
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
        description="Run a target-clipping diagnostic experiment for held-out sessions."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/target_clipping_experiment.json",
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
        "--clip-quantile",
        type=float,
        default=0.99,
        help="Upper quantile used to clip the training target.",
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


def evaluate_session(
    rows: list[dict],
    held_out_session: int,
    target_column: str,
    ridge_alpha: float,
    feature_groups: list[str],
    clip_quantile: float,
) -> dict:
    split = split_by_session(rows, held_out_session=held_out_session)
    filtered_train_rows, dropped_train_count = filter_rows_for_target(split.train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(split.test_rows, target_column)
    clipped_train_rows, clip_value = clip_rows_for_target(
        filtered_train_rows,
        target_column=target_column,
        upper_quantile=clip_quantile,
    )

    x_train, y_train = build_design_matrix(
        clipped_train_rows,
        target_column=target_column,
        feature_groups=feature_groups,
    )
    x_test, y_test = build_design_matrix(
        filtered_test_rows,
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
    errors = [float(prediction) - float(actual) for prediction, actual in zip(predictions, y_test)]
    abs_errors = [abs(error) for error in errors]
    sq_errors = [error * error for error in errors]

    top_rows: list[dict] = []
    for row, prediction, error in sorted(
        zip(filtered_test_rows, predictions, errors),
        key=lambda item: abs(float(item[2])),
        reverse=True,
    )[:10]:
        top_rows.append(
            {
                "session": int(row["session"]),
                "epoch": int(row["epoch"]),
                "task_environment": row["task_environment"],
                "tetrode": int(row["tetrode"]),
                "cell": int(row["cell"]),
                "actual": float(row[target_column]),
                "prediction": float(prediction),
                "error": float(error),
                "abs_error": abs(float(error)),
                "num_spikes": int(row["num_spikes"]),
            }
        )

    return {
        "session": held_out_session,
        "clip_value": clip_value,
        "train_count": len(filtered_train_rows),
        "test_count": len(filtered_test_rows),
        "dropped_train_count": dropped_train_count,
        "dropped_test_count": dropped_test_count,
        "mae": sum(abs_errors) / len(abs_errors),
        "rmse": (sum(sq_errors) / len(sq_errors)) ** 0.5,
        "top_residual_rows": top_rows,
    }


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    session_results = [
        evaluate_session(
            rows,
            held_out_session=session,
            target_column=args.target,
            ridge_alpha=args.ridge_alpha,
            feature_groups=args.feature_groups,
            clip_quantile=args.clip_quantile,
        )
        for session in args.sessions
    ]
    payload = {
        "target_column": args.target,
        "ridge_alpha": args.ridge_alpha,
        "feature_groups": args.feature_groups,
        "clip_quantile": args.clip_quantile,
        "sessions": session_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_rows = []
    top_rows = []
    for session_result in session_results:
        summary_rows.append(
            {
                key: value
                for key, value in session_result.items()
                if key != "top_residual_rows"
            }
        )
        top_rows.extend(session_result["top_residual_rows"])
    write_csv(output_path.with_suffix(".sessions.csv"), summary_rows)
    write_csv(output_path.with_suffix(".rows.csv"), top_rows)

    print(json.dumps(payload, indent=2))
    print(f"Wrote clipping experiment to {output_path}")
    print(f"Wrote session summary table to {output_path.with_suffix('.sessions.csv')}")
    print(f"Wrote residual row table to {output_path.with_suffix('.rows.csv')}")


if __name__ == "__main__":
    main()
