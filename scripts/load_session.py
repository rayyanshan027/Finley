from __future__ import annotations

import argparse
import json

from finley.config import load_config
from finley.data.session import load_session_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load HC-6 MATLAB files for one animal/session.")
    parser.add_argument("--config", default="configs/hc6.local.json", help="Path to JSON config.")
    parser.add_argument("--animal", default="Bon", help="Animal ID, for example Bon.")
    parser.add_argument("--session", type=int, default=3, help="Session number, for example 3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    loaded = load_session_files(config.dataset, args.animal, args.session)

    summary = {}
    for label, contents in loaded.items():
        keys = sorted(key for key in contents.keys() if not key.startswith("__"))
        summary[label] = {"variable_count": len(keys), "variables": keys[:25]}

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
