from __future__ import annotations

import argparse
import json
from pathlib import Path

from finley.config import load_config
from finley.data.session import summarize_session_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize parsed HC-6 session structure.")
    parser.add_argument("--config", default="configs/hc6.local.json", help="Path to JSON config.")
    parser.add_argument("--animal", default="Bon", help="Animal ID, for example Bon.")
    parser.add_argument("--session", type=int, default=3, help="Session number, for example 3.")
    parser.add_argument(
        "--output",
        default="data/inventory/hc6_session_summary.json",
        help="Path to JSON output summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = summarize_session_data(config.dataset, args.animal, args.session)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote parsed session summary to {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
