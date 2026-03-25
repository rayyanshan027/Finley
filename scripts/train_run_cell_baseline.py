from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.models.run_cell_baseline import (
    compute_metrics,
    filter_rows_by_environment,
    get_default_alpha_sweep_specs,
    get_available_feature_groups,
    load_model_table,
    run_leave_one_session_out,
    run_alpha_sweep,
    run_feature_ablations,
    split_by_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple baseline on the run-cell model table.")
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
        default="artifacts/run_cell_baseline_metrics.json",
        help="Path to JSON metrics output.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=100.0,
        help="Ridge regularization strength for the baseline.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        choices=get_available_feature_groups(),
        default=["movement_summaries"],
        help="Optional feature groups to include. Defaults to movement_summaries.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run the standard feature-group ablation suite and write a table of results.",
    )
    parser.add_argument(
        "--alpha-sweep",
        nargs="+",
        type=float,
        help="Run an alpha sweep for the default shortlist of feature-group settings.",
    )
    parser.add_argument(
        "--leave-one-session-out",
        action="store_true",
        help="Evaluate the baseline by holding out each session in turn and write per-session metrics.",
    )
    parser.add_argument(
        "--by-track",
        action="store_true",
        help="Run separate evaluations for TrackA and TrackB using the selected evaluation mode.",
    )
    return parser.parse_args()


def write_ablation_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode_count = sum(bool(value) for value in [args.ablation, args.alpha_sweep, args.leave_one_session_out])
    if mode_count > 1:
        raise ValueError("Use only one of --ablation, --alpha-sweep, or --leave-one-session-out.")
    if args.by_track and not args.leave_one_session_out:
        raise ValueError("--by-track currently requires --leave-one-session-out.")
    if args.alpha_sweep:
        results = run_alpha_sweep(
            rows,
            target_column=args.target,
            ridge_alphas=args.alpha_sweep,
            held_out_session=args.held_out_session,
            feature_group_specs=get_default_alpha_sweep_specs(),
        )
        result_rows = []
        for result in results:
            row = dict(result.__dict__)
            row["feature_groups"] = ",".join(result.feature_groups)
            result_rows.append(row)
        output_path.write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
        csv_path = output_path.with_suffix(".csv")
        write_ablation_csv(csv_path, result_rows)
        print(json.dumps(result_rows, indent=2))
        print(f"Wrote alpha sweep results to {output_path}")
        print(f"Wrote alpha sweep table to {csv_path}")
        return
    if args.leave_one_session_out:
        if args.by_track:
            payload: dict[str, object] = {}
            session_rows: list[dict] = []
            for environment in ["TrackA", "TrackB"]:
                track_rows = filter_rows_by_environment(rows, environment)
                metrics_by_session, summary = run_leave_one_session_out(
                    track_rows,
                    target_column=args.target,
                    ridge_alpha=args.ridge_alpha,
                    feature_groups=args.feature_groups,
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
            write_ablation_csv(csv_path, session_rows)
            print(json.dumps(payload, indent=2))
            print(f"Wrote track-specific leave-one-session-out results to {output_path}")
            print(f"Wrote track-specific leave-one-session-out table to {csv_path}")
            return
        metrics_by_session, summary = run_leave_one_session_out(
            rows,
            target_column=args.target,
            ridge_alpha=args.ridge_alpha,
            feature_groups=args.feature_groups,
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
        write_ablation_csv(csv_path, session_rows)
        print(json.dumps(payload, indent=2))
        print(f"Wrote leave-one-session-out results to {output_path}")
        print(f"Wrote leave-one-session-out table to {csv_path}")
        return
    if args.ablation:
        results = run_feature_ablations(
            rows,
            target_column=args.target,
            held_out_session=args.held_out_session,
            ridge_alpha=args.ridge_alpha,
        )
        result_rows = []
        for result in results:
            row = dict(result.__dict__)
            row["feature_groups"] = ",".join(result.feature_groups)
            result_rows.append(row)
        output_path.write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
        csv_path = output_path.with_suffix(".csv")
        write_ablation_csv(csv_path, result_rows)
        print(json.dumps(result_rows, indent=2))
        print(f"Wrote ablation results to {output_path}")
        print(f"Wrote ablation table to {csv_path}")
        return

    split = split_by_session(rows, held_out_session=args.held_out_session)
    metrics = compute_metrics(
        split.train_rows,
        split.test_rows,
        held_out_session=split.held_out_session,
        target_column=args.target,
        ridge_alpha=args.ridge_alpha,
        feature_groups=args.feature_groups,
    )
    output_path.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")

    print(json.dumps(metrics.__dict__, indent=2))
    print(f"Wrote baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
