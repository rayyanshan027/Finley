from __future__ import annotations

import argparse
import csv
from pathlib import Path

from finley.config import load_config
from finley.data.session import build_cell_rows_from_loaded, build_epoch_rows_from_loaded, load_session_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export flattened HC-6 session tables.")
    parser.add_argument("--config", default="configs/hc6.local.json", help="Path to JSON config.")
    parser.add_argument("--animal", default="Bon", help="Animal ID, for example Bon.")
    parser.add_argument("--session", type=int, default=3, help="Session number, for example 3.")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for generated CSV tables.",
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

    loaded = load_session_files(config.dataset, args.animal, args.session)
    cell_rows = build_cell_rows_from_loaded(loaded, args.animal, args.session)
    epoch_rows = build_epoch_rows_from_loaded(loaded, args.animal, args.session, cell_rows=cell_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epoch_path = output_dir / f"{args.animal.lower()}_session{args.session:02d}_epochs.csv"
    cell_path = output_dir / f"{args.animal.lower()}_session{args.session:02d}_cells.csv"

    write_csv(epoch_path, epoch_rows)
    write_csv(cell_path, cell_rows)

    print(f"Wrote epoch table to {epoch_path} ({len(epoch_rows)} rows)")
    print(f"Wrote cell table to {cell_path} ({len(cell_rows)} rows)")


if __name__ == "__main__":
    main()
