from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from finley.models.run_cell_baseline import (
    filter_rows_by_environment,
    get_available_feature_groups,
    list_sessions,
    load_model_table,
    resolve_feature_groups,
    split_by_session,
)
from finley.models.run_cell_nonlinear import build_feature_matrix, filter_rows_for_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scikit-learn gradient boosting baseline on the run-cell model table."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
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
        "--held-out-session",
        type=int,
        help="Optional held-out session. Defaults to the highest session id in the table.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/run_cell_sklearn_gbdt_metrics.json",
        help="Path to JSON metrics output.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=get_available_feature_groups(),
        default=["movement_summaries", "population_context", "cell_metadata"],
        help="Feature groups to include.",
    )
    parser.add_argument(
        "--leave-one-session-out",
        action="store_true",
        help="Evaluate by holding out each session in turn.",
    )
    parser.add_argument(
        "--by-track",
        action="store_true",
        help="Run separate evaluations for TrackA and TrackB using the selected evaluation mode.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate.")
    parser.add_argument("--max-iter", type=int, default=300, help="Number of boosting iterations.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum tree depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=8, help="Minimum samples per leaf.")
    parser.add_argument("--l2-regularization", type=float, default=0.0, help="L2 regularization.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_metrics(
    rows: list[dict],
    held_out_session: int,
    target_column: str,
    feature_groups: list[str],
    learning_rate: float,
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_seed: int,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    split = split_by_session(rows, held_out_session=held_out_session)
    filtered_train_rows, dropped_train_count = filter_rows_for_target(split.train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(split.test_rows, target_column)

    x_train = build_feature_matrix(filtered_train_rows, feature_groups=feature_groups)
    y_train = [float(row[target_column]) for row in filtered_train_rows]
    x_test = build_feature_matrix(filtered_test_rows, feature_groups=feature_groups)
    y_test = [float(row[target_column]) for row in filtered_test_rows]

    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        random_state=random_seed,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    errors = [float(prediction) - float(actual) for prediction, actual in zip(predictions, y_test)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {
        "train_count": len(filtered_train_rows),
        "test_count": len(filtered_test_rows),
        "dropped_train_count": dropped_train_count,
        "dropped_test_count": dropped_test_count,
        "feature_groups": feature_groups,
        "feature_count": len(x_train[0]) if x_train else 0,
        "held_out_session": held_out_session,
        "target_column": target_column,
        "mae": mae,
        "rmse": rmse,
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "random_seed": random_seed,
        "model_name": "HistGradientBoostingRegressor",
    }


def run_leave_one_session_out(
    rows: list[dict],
    target_column: str,
    feature_groups: list[str],
    learning_rate: float,
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    l2_regularization: float,
    random_seed: int,
) -> tuple[list[dict], dict]:
    sessions = list_sessions(rows)
    metrics_by_session = [
        compute_metrics(
            rows,
            held_out_session=session,
            target_column=target_column,
            feature_groups=feature_groups,
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_seed=random_seed,
        )
        for session in sessions
    ]
    summary = {
        "target_column": target_column,
        "feature_groups": feature_groups,
        "feature_count": metrics_by_session[0]["feature_count"],
        "session_count": len(metrics_by_session),
        "mean_mae": sum(metric["mae"] for metric in metrics_by_session) / len(metrics_by_session),
        "mean_rmse": sum(metric["rmse"] for metric in metrics_by_session) / len(metrics_by_session),
        "learning_rate": learning_rate,
        "max_iter": max_iter,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "l2_regularization": l2_regularization,
        "random_seed": random_seed,
        "model_name": "HistGradientBoostingRegressor",
    }
    return metrics_by_session, summary


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    feature_groups = resolve_feature_groups(args.feature_groups)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.by_track and not args.leave_one_session_out:
        raise ValueError("--by-track currently requires --leave-one-session-out.")

    if args.leave_one_session_out:
        if args.by_track:
            payload: dict[str, object] = {}
            session_rows: list[dict] = []
            for environment in ["TrackA", "TrackB"]:
                track_rows = filter_rows_by_environment(rows, environment)
                metrics_by_session, summary = run_leave_one_session_out(
                    track_rows,
                    target_column=args.target,
                    feature_groups=feature_groups,
                    learning_rate=args.learning_rate,
                    max_iter=args.max_iter,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    l2_regularization=args.l2_regularization,
                    random_seed=args.random_seed,
                )
                payload[environment] = {
                    "summary": summary,
                    "sessions": metrics_by_session,
                }
                for metric in metrics_by_session:
                    row = dict(metric)
                    row["feature_groups"] = ",".join(metric["feature_groups"])
                    row["task_environment"] = environment
                    session_rows.append(row)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            csv_path = output_path.with_suffix(".csv")
            write_csv(csv_path, session_rows)
            print(json.dumps(payload, indent=2))
            print(f"Wrote track-specific leave-one-session-out results to {output_path}")
            print(f"Wrote track-specific leave-one-session-out table to {csv_path}")
            return

        metrics_by_session, summary = run_leave_one_session_out(
            rows,
            target_column=args.target,
            feature_groups=feature_groups,
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            l2_regularization=args.l2_regularization,
            random_seed=args.random_seed,
        )
        payload = {
            "summary": summary,
            "sessions": metrics_by_session,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        csv_path = output_path.with_suffix(".csv")
        session_rows = []
        for metric in metrics_by_session:
            row = dict(metric)
            row["feature_groups"] = ",".join(metric["feature_groups"])
            session_rows.append(row)
        write_csv(csv_path, session_rows)
        print(json.dumps(payload, indent=2))
        print(f"Wrote leave-one-session-out results to {output_path}")
        print(f"Wrote leave-one-session-out table to {csv_path}")
        return

    held_out_session = args.held_out_session
    if held_out_session is None:
        held_out_session = max(list_sessions(rows))
    metrics = compute_metrics(
        rows,
        held_out_session=held_out_session,
        target_column=args.target,
        feature_groups=feature_groups,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        random_seed=args.random_seed,
    )
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote sklearn gradient boosting metrics to {output_path}")


if __name__ == "__main__":
    main()
