from __future__ import annotations

import argparse
import json
from pathlib import Path

from finley.models.run_cell_baseline import compute_metrics, load_model_table, split_by_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple baseline on the run-cell model table.")
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--target",
        default="log_num_spikes",
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
        default=1.0,
        help="Ridge regularization strength for the baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    split = split_by_session(rows, held_out_session=args.held_out_session)
    metrics = compute_metrics(
        split.train_rows,
        split.test_rows,
        held_out_session=split.held_out_session,
        target_column=args.target,
        ridge_alpha=args.ridge_alpha,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics.__dict__, indent=2), encoding="utf-8")

    print(json.dumps(metrics.__dict__, indent=2))
    print(f"Wrote baseline metrics to {output_path}")


if __name__ == "__main__":
    main()
