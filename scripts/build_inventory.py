from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.config import load_config
from finley.data.hc6 import scan_dataset, summarize_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a file inventory for the HC-6 dataset.")
    parser.add_argument(
        "--config",
        default="configs/hc6.local.json",
        help="Path to JSON config. Defaults to configs/hc6.local.json.",
    )
    return parser.parse_args()


def write_csv(records: list[dict], output_path: Path) -> None:
    fieldnames = [
        "animal",
        "relative_path",
        "extension",
        "size_bytes",
        "path_depth",
        "modality",
        "session",
        "top_level_dir",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    records = scan_dataset(config.dataset)
    summary = summarize_inventory(records)

    csv_path = Path(config.inventory.output_csv)
    json_path = Path(config.inventory.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    write_csv(records, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote inventory CSV to {csv_path}")
    print(f"Wrote inventory summary to {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
