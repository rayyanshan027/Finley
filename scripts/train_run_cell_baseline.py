from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.models.run_cell_baseline import (
    compute_metrics,
    get_default_alpha_sweep_specs,
    get_available_feature_groups,
    load_model_table,
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
        choices=["log_num_spikes", "num_spikes", "firing_rate_hz", "log_firing_rate_hz"],
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
        default=["task_context", "movement_summaries"],
        help="Optional feature groups to include. Defaults to task_context and movement_summaries.",
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
    if args.ablation and args.alpha_sweep:
        raise ValueError("Use either --ablation or --alpha-sweep, not both.")
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
