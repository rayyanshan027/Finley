from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.hard_sessions import build_hard_session_diagnostics
from finley.models.run_cell_baseline import load_model_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect hard sessions in the run-cell model table using quantiles and outlier rows."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/hard_session_diagnostics.json",
        help="Path to JSON output.",
    )
    parser.add_argument(
        "--hard-sessions",
        nargs="+",
        type=int,
        default=[6, 7, 9],
        help="Hard sessions to inspect together.",
    )
    parser.add_argument(
        "--easy-sessions",
        nargs="+",
        type=int,
        default=[3, 4, 8],
        help="Easy sessions to use as comparison.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top rows to keep per metric and hard session.",
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


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    diagnostics = build_hard_session_diagnostics(
        rows,
        hard_sessions=args.hard_sessions,
        easy_sessions=args.easy_sessions,
        top_n=args.top_n,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

    session_csv = output_path.with_suffix(".sessions.csv")
    track_csv = output_path.with_suffix(".session_tracks.csv")
    outlier_csv = output_path.with_suffix(".outliers.csv")
    write_csv(session_csv, diagnostics["session_summaries"])
    write_csv(track_csv, diagnostics["session_track_summaries"])
    write_csv(outlier_csv, diagnostics["outlier_rows"])

    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote hard-session diagnostics to {output_path}")
    print(f"Wrote session summary table to {session_csv}")
    print(f"Wrote session-track summary table to {track_csv}")
    print(f"Wrote outlier table to {outlier_csv}")


if __name__ == "__main__":
    main()
