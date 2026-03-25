from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.adaptation import (
    apply_unit_residual_offsets,
    fit_unit_residual_offsets,
    list_session_epochs,
    summarize_errors,
    summarize_unit_offset_drift,
    summarize_unit_overlap,
)
from finley.models.run_cell_baseline import load_model_table
from finley.models.run_cell_nonlinear import (
    TreeRegressorConfig,
    build_feature_matrix,
    filter_rows_for_target,
    fit_random_forest,
    predict_forest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether later adaptation epochs help or hurt per-unit residual calibration."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/adaptation_epoch_residual_diagnostic.json",
        help="Path to JSON output.",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        type=int,
        default=[6, 7, 9],
        help="Held-out sessions to inspect.",
    )
    parser.add_argument(
        "--target",
        default="log_firing_rate_hz",
        choices=[
            "log_num_spikes",
            "num_spikes",
            "firing_rate_hz",
            "log_firing_rate_hz",
            "session_centered_log_firing_rate_hz",
        ],
        help="Target column to predict.",
    )
    parser.add_argument(
        "--feature-groups",
        nargs="+",
        default=["movement_summaries", "population_context", "cell_metadata"],
        help="Feature groups to include.",
    )
    parser.add_argument(
        "--unit-residual-shrinkage",
        type=float,
        default=0.0,
        help="Shrinkage strength for per-unit residual offsets.",
    )
    parser.add_argument("--n-estimators", type=int, default=48, help="Number of trees in the forest.")
    parser.add_argument("--max-depth", type=int, default=6, help="Maximum depth of each tree.")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=8,
        help="Minimum samples per leaf.",
    )
    parser.add_argument(
        "--max-features",
        default="sqrt",
        help="Feature subsampling per split: sqrt, all, or a positive integer.",
    )
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed for deterministic fitting.")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_max_features(value: str) -> str | int:
    if value in {"sqrt", "all"}:
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--max-features must be sqrt, all, or a positive integer.")
    return parsed


def _rows_for_epochs(rows: list[dict], session: int, epochs: list[int]) -> list[dict]:
    epoch_set = set(epochs)
    return [row for row in rows if int(row["session"]) == session and int(row["epoch"]) in epoch_set]


def evaluate_epoch_subset(
    rows: list[dict],
    held_out_session: int,
    adaptation_epochs: list[int],
    evaluation_epochs: list[int],
    target_column: str,
    feature_groups: list[str],
    config: TreeRegressorConfig,
    shrinkage: float,
    label: str,
) -> tuple[dict, dict[tuple[int, int], float]]:
    train_rows = [
        row
        for row in rows
        if int(row["session"]) != held_out_session or int(row["epoch"]) in set(adaptation_epochs)
    ]
    test_rows = _rows_for_epochs(rows, held_out_session, evaluation_epochs)
    filtered_train_rows, dropped_train_count = filter_rows_for_target(train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(test_rows, target_column)
    filtered_adaptation_rows = [
        row for row in filtered_train_rows if int(row["session"]) == held_out_session
    ]

    x_train = build_feature_matrix(filtered_train_rows, feature_groups=feature_groups)
    y_train = [float(row[target_column]) for row in filtered_train_rows]
    x_test = build_feature_matrix(filtered_test_rows, feature_groups=feature_groups)
    y_test = [float(row[target_column]) for row in filtered_test_rows]
    forest = fit_random_forest(x_train, y_train, config=config)
    base_predictions = predict_forest(x_test, forest)

    x_adaptation = build_feature_matrix(filtered_adaptation_rows, feature_groups=feature_groups)
    adaptation_predictions = predict_forest(x_adaptation, forest)
    adaptation_actuals = [float(row[target_column]) for row in filtered_adaptation_rows]
    adaptation_residuals = [
        actual - prediction
        for prediction, actual in zip(adaptation_predictions, adaptation_actuals)
    ]
    unit_offsets = fit_unit_residual_offsets(
        filtered_adaptation_rows,
        adaptation_residuals,
        shrinkage=shrinkage,
    )
    corrected_predictions, corrected_row_count = apply_unit_residual_offsets(
        filtered_test_rows,
        base_predictions,
        unit_offsets,
    )
    errors = [prediction - actual for prediction, actual in zip(corrected_predictions, y_test)]
    overlap_summary = summarize_unit_overlap(filtered_adaptation_rows, filtered_test_rows)
    return (
        {
            "held_out_session": held_out_session,
            "subset_label": label,
            "adaptation_epochs": adaptation_epochs,
            "evaluation_epochs": evaluation_epochs,
            "train_count": len(filtered_train_rows),
            "test_count": len(filtered_test_rows),
            "dropped_train_count": dropped_train_count,
            "dropped_test_count": dropped_test_count,
            "unit_residual_shrinkage": shrinkage,
            "unit_residual_offset_count": len(unit_offsets),
            "corrected_evaluation_row_count": corrected_row_count,
            **overlap_summary,
            **summarize_errors(errors),
        },
        unit_offsets,
    )


def main() -> None:
    args = parse_args()
    rows = load_model_table(args.input)
    config = TreeRegressorConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=parse_max_features(args.max_features),
        random_seed=args.random_seed,
    )

    results: list[dict] = []
    drift_rows: list[dict] = []
    for session in args.sessions:
        session_epochs = list_session_epochs(rows, session)
        if len(session_epochs) < 3:
            raise ValueError(
                f"Session {session} needs at least three epochs for this diagnostic; found {session_epochs}."
            )

        first_epoch = session_epochs[0]
        second_epoch = session_epochs[1]
        later_epochs = session_epochs[2:]

        first_result, first_offsets = evaluate_epoch_subset(
            rows,
            held_out_session=session,
            adaptation_epochs=[first_epoch],
            evaluation_epochs=later_epochs,
            target_column=args.target,
            feature_groups=args.feature_groups,
            config=config,
            shrinkage=args.unit_residual_shrinkage,
            label="first_epoch_only",
        )
        second_result, second_offsets = evaluate_epoch_subset(
            rows,
            held_out_session=session,
            adaptation_epochs=[second_epoch],
            evaluation_epochs=later_epochs,
            target_column=args.target,
            feature_groups=args.feature_groups,
            config=config,
            shrinkage=args.unit_residual_shrinkage,
            label="second_epoch_only",
        )
        pooled_result, pooled_offsets = evaluate_epoch_subset(
            rows,
            held_out_session=session,
            adaptation_epochs=[first_epoch, second_epoch],
            evaluation_epochs=later_epochs,
            target_column=args.target,
            feature_groups=args.feature_groups,
            config=config,
            shrinkage=args.unit_residual_shrinkage,
            label="first_plus_second_epochs",
        )
        results.extend([first_result, second_result, pooled_result])

        drift_rows.append(
            {
                "held_out_session": session,
                "first_epoch": first_epoch,
                "second_epoch": second_epoch,
                "evaluation_epochs": later_epochs,
                **summarize_unit_offset_drift(first_offsets, second_offsets),
                "first_epoch_offset_count": len(first_offsets),
                "second_epoch_offset_count": len(second_offsets),
                "pooled_offset_count": len(pooled_offsets),
            }
        )

    payload = {
        "target_column": args.target,
        "feature_groups": args.feature_groups,
        "sessions": args.sessions,
        "unit_residual_shrinkage": args.unit_residual_shrinkage,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "max_features": config.max_features,
        "random_seed": config.random_seed,
        "results": results,
        "offset_drift": drift_rows,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), results)
    write_csv(output_path.with_name(f"{output_path.stem}_offset_drift.csv"), drift_rows)

    print(json.dumps(payload, indent=2))
    print(f"Wrote adaptation epoch residual diagnostic to {output_path}")
    print(f"Wrote adaptation epoch residual table to {output_path.with_suffix('.csv')}")
    print(
        "Wrote adaptation epoch residual drift table to "
        f"{output_path.with_name(f'{output_path.stem}_offset_drift.csv')}"
    )


if __name__ == "__main__":
    main()
