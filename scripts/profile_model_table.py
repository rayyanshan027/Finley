from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.session_profile import summarize_model_table_by_session
from finley.models.run_cell_baseline import load_model_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the run-cell model table by session.")
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/run_cell_session_profiles.json",
        help="Path to JSON output.",
    )
    parser.add_argument(
        "--hard-sessions",
        nargs="+",
        type=int,
        help="Optional list of harder sessions to summarize together.",
    )
    parser.add_argument(
        "--easy-sessions",
        nargs="+",
        type=int,
        help="Optional list of easier sessions to summarize together.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    summary = summarize_model_table_by_session(
        rows,
        hard_sessions=args.hard_sessions,
        easy_sessions=args.easy_sessions,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    session_profiles = summary["session_profiles"]
    csv_path = output_path.with_suffix(".csv")
    write_csv(csv_path, session_profiles)

    print(json.dumps(summary, indent=2))
    print(f"Wrote session profiles to {output_path}")
    print(f"Wrote session profile table to {csv_path}")


if __name__ == "__main__":
    main()
