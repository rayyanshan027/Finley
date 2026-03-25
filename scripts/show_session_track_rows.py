from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_COLUMNS = [
    "session",
    "task_environment",
    "row_count",
    "epoch_count",
    "log_firing_rate_hz_mean",
    "log_firing_rate_hz_std",
    "mean_speed_mean",
    "std_speed_mean",
    "speed_q75_mean",
    "mean_abs_accel_mean",
    "spike_cell_count_mean",
    "spike_event_rows_epoch_mean",
    "num_spikes_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show selected rows from the session-track profile CSV.")
    parser.add_argument(
        "--input",
        default="artifacts/run_cell_session_profiles_tracks.csv",
        help="Path to the session-track profile CSV.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[4, 6, 7, 9],
        help="Sessions to include.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=DEFAULT_COLUMNS,
        help="Columns to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    target_sessions = set(args.sessions)
    filtered_rows = []
    for row in rows:
        if int(row["session"]) not in target_sessions:
            continue
        filtered_rows.append({column: row.get(column, "") for column in args.columns})

    print(json.dumps(filtered_rows, indent=2))


if __name__ == "__main__":
    main()
