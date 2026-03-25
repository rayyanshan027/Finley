from __future__ import annotations

import argparse
import csv
from pathlib import Path

from finley.config import load_config
from finley.data.session import (
    build_cell_rows_from_loaded,
    build_epoch_rows_from_loaded,
    build_run_cell_model_rows,
    list_available_sessions,
    load_session_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a run-only HC-6 modeling table.")
    parser.add_argument("--config", default="configs/hc6.local.json", help="Path to JSON config.")
    parser.add_argument("--animal", default="Bon", help="Animal ID, for example Bon.")
    parser.add_argument(
        "--output",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to output CSV.",
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
    config = load_config(args.config)
    sessions = list_available_sessions(config.dataset, args.animal)
    if not sessions:
        raise FileNotFoundError(f"No sessions found for animal {args.animal}")

    epoch_rows: list[dict] = []
    cell_rows: list[dict] = []
    for session in sessions:
        loaded = load_session_files(config.dataset, args.animal, session)
        session_cell_rows = build_cell_rows_from_loaded(loaded, args.animal, session)
        session_epoch_rows = build_epoch_rows_from_loaded(
            loaded,
            args.animal,
            session,
            cell_rows=session_cell_rows,
        )
        epoch_rows.extend(session_epoch_rows)
        cell_rows.extend(session_cell_rows)

    model_rows = build_run_cell_model_rows(epoch_rows, cell_rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, model_rows)

    print(f"Wrote run-cell model table to {output_path} ({len(model_rows)} rows)")


if __name__ == "__main__":
    main()
