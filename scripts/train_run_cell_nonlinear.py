from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.models.run_cell_nonlinear import (
    TreeRegressorConfig,
    compute_nonlinear_metrics,
    filter_rows_by_environment,
    get_available_feature_groups,
    load_model_table,
    run_leave_one_session_out_nonlinear,
    split_by_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pure-Python nonlinear baseline on the run-cell model table."
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
        default="artifacts/run_cell_nonlinear_metrics.json",
        help="Path to JSON metrics output.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=get_available_feature_groups(),
        default=["movement_summaries"],
        help="Feature groups to include. Defaults to movement_summaries.",
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = TreeRegressorConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        random_seed=args.random_seed,
    )

    if args.by_track and not args.leave_one_session_out:
        raise ValueError("--by-track currently requires --leave-one-session-out.")

    if args.leave_one_session_out:
        if args.by_track:
            payload: dict[str, object] = {}
            session_rows: list[dict] = []
            for environment in ["TrackA", "TrackB"]:
                track_rows = filter_rows_by_environment(rows, environment)
                metrics_by_session, summary = run_leave_one_session_out_nonlinear(
                    track_rows,
                    target_column=args.target,
                    feature_groups=args.feature_groups,
                    config=config,
                )
                payload[environment] = {
                    "summary": dict(summary.__dict__),
                    "sessions": [dict(metric.__dict__) for metric in metrics_by_session],
                }
                for metric in metrics_by_session:
                    row = dict(metric.__dict__)
                    row["feature_groups"] = ",".join(metric.feature_groups)
                    row["task_environment"] = environment
                    session_rows.append(row)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            csv_path = output_path.with_suffix(".csv")
            write_csv(csv_path, session_rows)
            print(json.dumps(payload, indent=2))
            print(f"Wrote track-specific leave-one-session-out results to {output_path}")
            print(f"Wrote track-specific leave-one-session-out table to {csv_path}")
            return

        metrics_by_session, summary = run_leave_one_session_out_nonlinear(
            rows,
            target_column=args.target,
            feature_groups=args.feature_groups,
            config=config,
        )
        payload = {
            "summary": dict(summary.__dict__),
            "sessions": [dict(metric.__dict__) for metric in metrics_by_session],
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        csv_path = output_path.with_suffix(".csv")
        session_rows = []
        for metric in metrics_by_session:
            row = dict(metric.__dict__)
            row["feature_groups"] = ",".join(metric.feature_groups)
            session_rows.append(row)
        write_csv(csv_path, session_rows)
        print(json.dumps(payload, indent=2))
        print(f"Wrote leave-one-session-out results to {output_path}")
        print(f"Wrote leave-one-session-out table to {csv_path}")
        return

    split = split_by_session(rows, held_out_session=args.held_out_session)
    metrics = compute_nonlinear_metrics(
        split.train_rows,
        split.test_rows,
        held_out_session=split.held_out_session,
        target_column=args.target,
        feature_groups=args.feature_groups,
        config=config,
    )
    output_path.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")
    print(json.dumps(metrics.__dict__, indent=2))
    print(f"Wrote nonlinear baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
