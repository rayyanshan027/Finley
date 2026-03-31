from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all-animal LOSO benchmark outputs for nonlinear and XGBoost runs."
    )
    parser.add_argument(
        "--input-dir",
        default="artifacts/all_animals",
        help="Directory containing per-animal benchmark JSON outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV output path for the summary table.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["summary"])


def build_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for nonlinear_path in sorted(input_dir.glob("*_nonlinear_loso.json")):
        animal = nonlinear_path.name.removesuffix("_nonlinear_loso.json")
        xgboost_path = input_dir / f"{animal}_xgboost_loso.json"
        if not xgboost_path.exists():
            continue

        nonlinear = load_summary(nonlinear_path)
        xgboost = load_summary(xgboost_path)
        rows.append(
            {
                "animal": animal,
                "session_count": int(nonlinear["session_count"]),
                "nonlinear_mae": float(nonlinear["mean_mae"]),
                "nonlinear_rmse": float(nonlinear["mean_rmse"]),
                "xgboost_mae": float(xgboost["mean_mae"]),
                "xgboost_rmse": float(xgboost["mean_rmse"]),
                "winner": (
                    "nonlinear"
                    if float(nonlinear["mean_mae"]) < float(xgboost["mean_mae"])
                    else "xgboost"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    print("animal,session_count,nonlinear_mae,nonlinear_rmse,xgboost_mae,xgboost_rmse,winner")
    for row in rows:
        print(
            f"{row['animal']},{row['session_count']},"
            f"{row['nonlinear_mae']:.4f},{row['nonlinear_rmse']:.4f},"
            f"{row['xgboost_mae']:.4f},{row['xgboost_rmse']:.4f},{row['winner']}"
        )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = build_rows(input_dir)
    print_table(rows)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_path, rows)
        print(f"Wrote all-animal benchmark summary to {output_path}")


if __name__ == "__main__":
    main()
