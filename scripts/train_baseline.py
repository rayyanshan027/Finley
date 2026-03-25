from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.config import load_config
from finley.models.baseline import run_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a starter baseline over the HC-6 inventory.")
    parser.add_argument(
        "--config",
        default="configs/hc6.local.json",
        help="Path to JSON config. Defaults to configs/hc6.local.json.",
    )
    return parser.parse_args()


def load_inventory(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = []
        for row in reader:
            row["size_bytes"] = int(row["size_bytes"])
            row["path_depth"] = int(row["path_depth"])
            row["session"] = int(row["session"]) if row["session"] else None
            records.append(row)
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    inventory_path = Path(config.inventory.output_csv)
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory CSV does not exist at {inventory_path}. Run scripts/build_inventory.py first."
        )

    records = load_inventory(inventory_path)
    if len(records) < config.training.min_rows:
        raise ValueError(
            f"Inventory has {len(records)} rows, below min_rows={config.training.min_rows}. "
            "Relax the config or scan more data."
        )

    result = run_baseline(records, config.training.feature_columns)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
