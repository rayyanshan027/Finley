from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from finley.analysis.adaptation import (
    apply_unit_residual_offsets,
    fit_unit_residual_offsets,
    split_session_adaptation_rows,
    summarize_errors,
    summarize_unit_overlap,
)
from finley.models.run_cell_baseline import load_model_table
from finley.models.run_cell_nonlinear import (
    TreeRegressorConfig,
    build_feature_matrix,
    filter_rows_for_target,
    fit_session_unit_feature_encoder,
    fit_random_forest,
    predict_forest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a held-out-session adaptation experiment using epoch-level adaptation splits."
    )
    parser.add_argument(
        "--input",
        default="data/processed/bon_run_cell_model_table.csv",
        help="Path to run-cell model table CSV.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/session_adaptation_experiment.json",
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
        "--adaptation-epochs",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="How many earliest epochs from the held-out session to add to training.",
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
        "--model-variants",
        nargs="+",
        default=[
            "baseline",
            "session_unit_identity",
            "baseline_plus_unit_residual",
            "baseline_plus_latest_unit_residual",
        ],
        choices=[
            "baseline",
            "session_unit_identity",
            "baseline_plus_unit_residual",
            "baseline_plus_latest_unit_residual",
        ],
        help="Adaptive model variants to evaluate.",
    )
    parser.add_argument(
        "--unit-residual-shrinkage-values",
        nargs="+",
        type=float,
        default=[0.0],
        help="Shrinkage strengths to compare for per-unit residual offsets.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=48,
        help="Number of trees in the forest.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum depth of each tree.",
    )
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
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for deterministic fitting.",
    )
    return parser.parse_args()


def parse_max_features(value: str) -> str | int:
    if value in {"sqrt", "all"}:
        return value
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--max-features must be sqrt, all, or a positive integer.")
    return parsed


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_unit_residual_shrinkages(model_variant: str, shrinkage_values: list[float]) -> list[float]:
    if any(value < 0 for value in shrinkage_values):
        raise ValueError("--unit-residual-shrinkage-values must all be non-negative.")
    if model_variant in {"baseline_plus_unit_residual", "baseline_plus_latest_unit_residual"}:
        return list(shrinkage_values)
    return [shrinkage_values[0]]


def resolve_residual_adaptation_rows(
    model_variant: str,
    adaptation_rows: list[dict],
) -> list[dict]:
    if model_variant != "baseline_plus_latest_unit_residual":
        return list(adaptation_rows)
    if not adaptation_rows:
        return []
    latest_epoch = max(int(row["epoch"]) for row in adaptation_rows)
    return [row for row in adaptation_rows if int(row["epoch"]) == latest_epoch]


def evaluate_adaptation_setting(
    rows: list[dict],
    held_out_session: int,
    adaptation_epoch_count: int,
    target_column: str,
    feature_groups: list[str],
    config: TreeRegressorConfig,
    model_variant: str,
    unit_residual_shrinkage: float,
) -> dict:
    train_rows, test_rows, adaptation_epochs, evaluation_epochs = split_session_adaptation_rows(
        rows,
        held_out_session=held_out_session,
        adaptation_epoch_count=adaptation_epoch_count,
    )
    filtered_train_rows, dropped_train_count = filter_rows_for_target(train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(test_rows, target_column)
    filtered_adaptation_rows = [
        row for row in filtered_train_rows if int(row["session"]) == held_out_session
    ]
    overlap_summary = summarize_unit_overlap(filtered_adaptation_rows, filtered_test_rows)
    session_unit_encoder = None
    if model_variant == "session_unit_identity":
        session_unit_encoder = fit_session_unit_feature_encoder(filtered_adaptation_rows)

    x_train = build_feature_matrix(
        filtered_train_rows,
        feature_groups=feature_groups,
        session_unit_encoder=session_unit_encoder,
    )
    y_train = [float(row[target_column]) for row in filtered_train_rows]
    x_test = build_feature_matrix(
        filtered_test_rows,
        feature_groups=feature_groups,
        session_unit_encoder=session_unit_encoder,
    )
    y_test = [float(row[target_column]) for row in filtered_test_rows]

    forest = fit_random_forest(x_train, y_train, config=config)
    predictions = predict_forest(x_test, forest)
    corrected_row_count = 0
    unit_residual_offset_count = 0
    residual_adaptation_rows = resolve_residual_adaptation_rows(
        model_variant,
        filtered_adaptation_rows,
    )
    if model_variant in {"baseline_plus_unit_residual", "baseline_plus_latest_unit_residual"}:
        x_adaptation = build_feature_matrix(
            residual_adaptation_rows,
            feature_groups=feature_groups,
        )
        adaptation_predictions = predict_forest(x_adaptation, forest)
        adaptation_residuals = [
            float(actual) - float(prediction)
            for prediction, actual in zip(
                adaptation_predictions,
                [row[target_column] for row in residual_adaptation_rows],
            )
        ]
        unit_offsets = fit_unit_residual_offsets(
            residual_adaptation_rows,
            adaptation_residuals,
            shrinkage=unit_residual_shrinkage,
        )
        predictions, corrected_row_count = apply_unit_residual_offsets(
            filtered_test_rows,
            predictions,
            unit_offsets,
        )
        unit_residual_offset_count = len(unit_offsets)
    errors = [float(prediction) - float(actual) for prediction, actual in zip(predictions, y_test)]
    summary = summarize_errors(errors)
    return {
        "held_out_session": held_out_session,
        "model_variant": model_variant,
        "uses_session_unit_identity": model_variant == "session_unit_identity",
        "uses_unit_residual_correction": model_variant
        in {"baseline_plus_unit_residual", "baseline_plus_latest_unit_residual"},
        "uses_latest_unit_residual_only": model_variant == "baseline_plus_latest_unit_residual",
        "adaptation_epoch_count": adaptation_epoch_count,
        "adaptation_epochs": adaptation_epochs,
        "residual_adaptation_epochs": sorted(
            {int(row["epoch"]) for row in residual_adaptation_rows}
        ),
        "evaluation_epochs": evaluation_epochs,
        "train_count": len(filtered_train_rows),
        "test_count": len(filtered_test_rows),
        "dropped_train_count": dropped_train_count,
        "dropped_test_count": dropped_test_count,
        "session_unit_feature_count": (
            session_unit_encoder.feature_count if session_unit_encoder is not None else 0
        ),
        "unit_residual_shrinkage": unit_residual_shrinkage,
        "unit_residual_offset_count": unit_residual_offset_count,
        "corrected_evaluation_row_count": corrected_row_count,
        **overlap_summary,
        **summary,
    }


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
    for session in args.sessions:
        for adaptation_epoch_count in args.adaptation_epochs:
            for model_variant in args.model_variants:
                for unit_residual_shrinkage in resolve_unit_residual_shrinkages(
                    model_variant,
                    args.unit_residual_shrinkage_values,
                ):
                    results.append(
                        evaluate_adaptation_setting(
                            rows,
                            held_out_session=session,
                            adaptation_epoch_count=adaptation_epoch_count,
                            target_column=args.target,
                            feature_groups=args.feature_groups,
                            config=config,
                            model_variant=model_variant,
                            unit_residual_shrinkage=unit_residual_shrinkage,
                        )
                    )

    payload = {
        "target_column": args.target,
        "feature_groups": args.feature_groups,
        "model_variants": args.model_variants,
        "sessions": args.sessions,
        "adaptation_epochs": args.adaptation_epochs,
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_leaf": config.min_samples_leaf,
        "max_features": config.max_features,
        "random_seed": config.random_seed,
        "unit_residual_shrinkage_values": args.unit_residual_shrinkage_values,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), results)

    print(json.dumps(payload, indent=2))
    print(f"Wrote session adaptation experiment to {output_path}")
    print(f"Wrote session adaptation table to {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
