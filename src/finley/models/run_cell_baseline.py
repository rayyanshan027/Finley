from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path

FEATURE_GROUP_ORDER = [
    "task_context",
    "movement_summaries",
    "population_context",
    "cell_metadata",
]

FEATURE_GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "task_context": ("track_a", "track_b", "task_exposure", "task_experimentday"),
    "movement_summaries": (
        "pos_rows",
        "epoch_duration_sec",
        "mean_speed",
        "std_speed",
        "max_speed",
        "mean_accel",
        "std_accel",
        "mean_abs_accel",
        "max_abs_accel",
        "speed_q25",
        "speed_q50",
        "speed_q75",
        "moving_fraction",
        "fast_fraction",
        "path_length",
        "step_length_mean",
        "step_length_max",
        "x_range",
        "y_range",
    ),
    "population_context": (
        "rawpos_rows",
        "spike_tetrode_count",
        "other_epoch_cells",
        "other_epoch_spikes",
    ),
    "cell_metadata": ("tetrode", "depth", "spikewidth"),
}


@dataclass(frozen=True)
class TrainTestSplit:
    train_rows: list[dict]
    test_rows: list[dict]
    held_out_session: int


@dataclass(frozen=True)
class RegressionMetrics:
    train_count: int
    test_count: int
    dropped_train_count: int
    dropped_test_count: int
    feature_groups: list[str]
    feature_count: int
    held_out_session: int
    target_column: str
    mae: float
    rmse: float


@dataclass(frozen=True)
class FeatureScaler:
    means: list[float]
    scales: list[float]


@dataclass(frozen=True)
class FeatureAblationResult:
    name: str
    feature_groups: list[str]
    feature_count: int
    train_count: int
    test_count: int
    dropped_train_count: int
    dropped_test_count: int
    held_out_session: int
    target_column: str
    mae: float
    rmse: float


@dataclass(frozen=True)
class AlphaSweepResult:
    name: str
    feature_groups: list[str]
    feature_count: int
    ridge_alpha: float
    train_count: int
    test_count: int
    dropped_train_count: int
    dropped_test_count: int
    held_out_session: int
    target_column: str
    mae: float
    rmse: float


@dataclass(frozen=True)
class CrossSessionSummary:
    target_column: str
    feature_groups: list[str]
    feature_count: int
    ridge_alpha: float
    session_count: int
    mean_mae: float
    mean_rmse: float


def list_sessions(rows: list[dict]) -> list[int]:
    return sorted({int(row["session"]) for row in rows})


def load_model_table(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        int_fields = {
            "session",
            "epoch",
            "pos_rows",
            "rawpos_rows",
            "spike_tetrode_count",
            "spike_cell_count",
            "spike_event_rows_epoch",
            "tetrode",
            "cell",
            "num_spikes",
        }
        float_fields = {
            "task_exposure",
            "task_experimentday",
            "epoch_duration_sec",
            "mean_speed",
            "std_speed",
            "max_speed",
            "mean_accel",
            "std_accel",
            "mean_abs_accel",
            "max_abs_accel",
            "speed_q25",
            "speed_q50",
            "speed_q75",
            "moving_fraction",
            "fast_fraction",
            "path_length",
            "step_length_mean",
            "step_length_max",
            "x_range",
            "y_range",
            "mean_dir",
            "dir_std",
            "depth",
            "spikewidth",
            "firing_rate_hz",
            "log_num_spikes",
            "log_firing_rate_hz",
            "session_centered_log_firing_rate_hz",
        }
        for row in reader:
            parsed = dict(row)
            for field in int_fields:
                if field in parsed:
                    parsed[field] = int(parsed[field])
            for field in float_fields:
                if field in parsed:
                    parsed[field] = float(parsed[field]) if parsed[field] != "" else None
            rows.append(parsed)
    return rows


def split_by_session(rows: list[dict], held_out_session: int | None = None) -> TrainTestSplit:
    if not rows:
        raise ValueError("Model table is empty.")
    sessions = sorted({int(row["session"]) for row in rows})
    chosen = held_out_session if held_out_session is not None else sessions[-1]
    train_rows = [row for row in rows if int(row["session"]) != chosen]
    test_rows = [row for row in rows if int(row["session"]) == chosen]
    if not train_rows or not test_rows:
        raise ValueError(f"Unable to create train/test split with held_out_session={chosen}.")
    return TrainTestSplit(train_rows=train_rows, test_rows=test_rows, held_out_session=chosen)


def get_available_feature_groups() -> list[str]:
    return list(FEATURE_GROUP_ORDER)


def resolve_feature_groups(feature_groups: list[str] | None = None) -> list[str]:
    if feature_groups is None:
        return list(FEATURE_GROUP_ORDER)
    unknown = [group for group in feature_groups if group not in FEATURE_GROUP_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown feature groups: {unknown}")
    return list(feature_groups)


def _feature_map(row: dict) -> dict[str, float]:
    environment = row["task_environment"]
    track_a = 1.0 if environment == "TrackA" else 0.0
    track_b = 1.0 if environment == "TrackB" else 0.0
    other_epoch_spikes = max(int(row["spike_event_rows_epoch"]) - int(row["num_spikes"]), 0)
    other_epoch_cells = max(int(row["spike_cell_count"]) - 1, 0)

    return {
        "track_a": track_a,
        "track_b": track_b,
        "task_exposure": float(row["task_exposure"] or 0.0),
        "task_experimentday": float(row["task_experimentday"] or 0.0),
        "pos_rows": float(row["pos_rows"]),
        "epoch_duration_sec": float(row["epoch_duration_sec"] or 0.0),
        "mean_speed": float(row["mean_speed"] or 0.0),
        "std_speed": float(row["std_speed"] or 0.0),
        "max_speed": float(row["max_speed"] or 0.0),
        "mean_accel": float(row.get("mean_accel") or 0.0),
        "std_accel": float(row.get("std_accel") or 0.0),
        "mean_abs_accel": float(row.get("mean_abs_accel") or 0.0),
        "max_abs_accel": float(row.get("max_abs_accel") or 0.0),
        "speed_q25": float(row["speed_q25"] or 0.0),
        "speed_q50": float(row["speed_q50"] or 0.0),
        "speed_q75": float(row["speed_q75"] or 0.0),
        "moving_fraction": float(row["moving_fraction"] or 0.0),
        "fast_fraction": float(row["fast_fraction"] or 0.0),
        "path_length": float(row["path_length"] or 0.0),
        "step_length_mean": float(row["step_length_mean"] or 0.0),
        "step_length_max": float(row["step_length_max"] or 0.0),
        "x_range": float(row["x_range"] or 0.0),
        "y_range": float(row["y_range"] or 0.0),
        "rawpos_rows": float(row["rawpos_rows"]),
        "spike_tetrode_count": float(row["spike_tetrode_count"]),
        "other_epoch_cells": float(other_epoch_cells),
        "other_epoch_spikes": float(other_epoch_spikes),
        "tetrode": float(row["tetrode"]),
        "depth": float(row["depth"] or 0.0),
        "spikewidth": float(row["spikewidth"] or 0.0),
    }


def _feature_vector(row: dict, feature_groups: list[str] | None = None) -> list[float]:
    groups = resolve_feature_groups(feature_groups)
    feature_map = _feature_map(row)
    vector = [1.0]
    for group in groups:
        vector.extend(feature_map[column] for column in FEATURE_GROUP_COLUMNS[group])
    return vector


def get_feature_count(feature_groups: list[str] | None = None) -> int:
    groups = resolve_feature_groups(feature_groups)
    return 1 + sum(len(FEATURE_GROUP_COLUMNS[group]) for group in groups)


def build_design_matrix(
    rows: list[dict],
    target_column: str,
    feature_groups: list[str] | None = None,
) -> tuple[list[list[float]], list[float]]:
    if not rows:
        raise ValueError("Cannot build design matrix from empty rows.")
    missing_target_rows = [index for index, row in enumerate(rows) if row.get(target_column) is None]
    if missing_target_rows:
        raise ValueError(
            f"Target column {target_column} contains missing values in "
            f"{len(missing_target_rows)} row(s). Filter rows before training."
        )
    x = [_feature_vector(row, feature_groups=feature_groups) for row in rows]
    y = [float(row[target_column]) for row in rows]
    return x, y


def filter_rows_for_target(rows: list[dict], target_column: str) -> tuple[list[dict], int]:
    filtered = [row for row in rows if row.get(target_column) is not None]
    return filtered, len(rows) - len(filtered)


def fit_feature_scaler(x_train: list[list[float]]) -> FeatureScaler:
    if not x_train:
        raise ValueError("Cannot fit scaler on empty training data.")
    feature_count = len(x_train[0])
    means: list[float] = []
    scales: list[float] = []
    for column_index in range(feature_count):
        column = [row[column_index] for row in x_train]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance)
        if scale < 1e-8:
            scale = 1.0
        means.append(mean)
        scales.append(scale)

    # Preserve the intercept term as a constant 1 instead of standardizing it away.
    means[0] = 0.0
    scales[0] = 1.0
    return FeatureScaler(means=means, scales=scales)


def apply_feature_scaler(x: list[list[float]], scaler: FeatureScaler) -> list[list[float]]:
    transformed: list[list[float]] = []
    for row in x:
        transformed.append(
            [
                (value - mean) / scale
                for value, mean, scale in zip(row, scaler.means, scaler.scales)
            ]
        )
    return transformed


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(column) for column in zip(*matrix)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    b_t = _transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in b_t] for row in a]


def _invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    augmented = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("Singular matrix encountered while fitting baseline.")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

        pivot = augmented[col][col]
        augmented[col] = [value / pivot for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                current - factor * pivot_value
                for current, pivot_value in zip(augmented[row], augmented[col])
            ]

    return [row[n:] for row in augmented]


def fit_ridge_regression(x_train: list[list[float]], y_train: list[float], alpha: float = 1.0) -> list[float]:
    xt = _transpose(x_train)
    xtx = _matmul(xt, x_train)
    xty = [sum(feature * target for feature, target in zip(column, y_train)) for column in xt]
    xtx = [
        [
            value + (alpha if i == j and i != 0 else 0.0)
            for j, value in enumerate(row)
        ]
        for i, row in enumerate(xtx)
    ]
    inv_xtx = _invert_matrix(xtx)
    coefficients_matrix = _matmul(inv_xtx, [[value] for value in xty])
    return [row[0] for row in coefficients_matrix]


def predict(x: list[list[float]], coefficients: list[float]) -> list[float]:
    return [sum(feature * weight for feature, weight in zip(row, coefficients)) for row in x]


def compute_metrics(
    train_rows: list[dict],
    test_rows: list[dict],
    held_out_session: int,
    target_column: str,
    ridge_alpha: float = 1.0,
    feature_groups: list[str] | None = None,
) -> RegressionMetrics:
    resolved_feature_groups = resolve_feature_groups(feature_groups)
    filtered_train_rows, dropped_train_count = filter_rows_for_target(train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(test_rows, target_column)
    if not filtered_train_rows or not filtered_test_rows:
        raise ValueError(
            f"No usable rows remain for target {target_column} after filtering missing targets."
        )
    x_train, y_train = build_design_matrix(
        filtered_train_rows,
        target_column=target_column,
        feature_groups=resolved_feature_groups,
    )
    x_test, y_test = build_design_matrix(
        filtered_test_rows,
        target_column=target_column,
        feature_groups=resolved_feature_groups,
    )
    scaler = fit_feature_scaler(x_train)
    x_train_scaled = apply_feature_scaler(x_train, scaler)
    x_test_scaled = apply_feature_scaler(x_test, scaler)
    coefficients = fit_ridge_regression(x_train_scaled, y_train, alpha=ridge_alpha)
    predictions = predict(x_test_scaled, coefficients)
    errors = [prediction - actual for prediction, actual in zip(predictions, y_test)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return RegressionMetrics(
        train_count=len(filtered_train_rows),
        test_count=len(filtered_test_rows),
        dropped_train_count=dropped_train_count,
        dropped_test_count=dropped_test_count,
        feature_groups=resolved_feature_groups,
        feature_count=get_feature_count(resolved_feature_groups),
        held_out_session=held_out_session,
        target_column=target_column,
        mae=mae,
        rmse=rmse,
    )


def run_feature_ablations(
    rows: list[dict],
    target_column: str,
    held_out_session: int | None = None,
    ridge_alpha: float = 1.0,
    ) -> list[FeatureAblationResult]:
    split = split_by_session(rows, held_out_session=held_out_session)
    ablation_specs: list[tuple[str, list[str]]] = [
        ("all", list(FEATURE_GROUP_ORDER)),
        *[(group, [group]) for group in FEATURE_GROUP_ORDER],
        *[
            (f"all_minus_{group}", [name for name in FEATURE_GROUP_ORDER if name != group])
            for group in FEATURE_GROUP_ORDER
        ],
    ]
    results: list[FeatureAblationResult] = []
    for name, feature_groups in ablation_specs:
        metrics = compute_metrics(
            split.train_rows,
            split.test_rows,
            held_out_session=split.held_out_session,
            target_column=target_column,
            ridge_alpha=ridge_alpha,
            feature_groups=feature_groups,
        )
        results.append(
            FeatureAblationResult(
                name=name,
                feature_groups=metrics.feature_groups,
                feature_count=metrics.feature_count,
                train_count=metrics.train_count,
                test_count=metrics.test_count,
                dropped_train_count=metrics.dropped_train_count,
                dropped_test_count=metrics.dropped_test_count,
                held_out_session=metrics.held_out_session,
                target_column=metrics.target_column,
                mae=metrics.mae,
                rmse=metrics.rmse,
            )
        )
    return results


def get_default_alpha_sweep_specs() -> list[tuple[str, list[str]]]:
    return [
        ("movement_summaries", ["movement_summaries"]),
        ("population_context", ["population_context"]),
        (
            "task_context_movement_summaries",
            ["task_context", "movement_summaries"],
        ),
        (
            "task_context_movement_summaries_cell_metadata",
            ["task_context", "movement_summaries", "cell_metadata"],
        ),
        (
            "task_context_population_context_cell_metadata",
            ["task_context", "population_context", "cell_metadata"],
        ),
    ]


def run_alpha_sweep(
    rows: list[dict],
    target_column: str,
    ridge_alphas: list[float],
    held_out_session: int | None = None,
    feature_group_specs: list[tuple[str, list[str]]] | None = None,
) -> list[AlphaSweepResult]:
    if not ridge_alphas:
        raise ValueError("Alpha sweep requires at least one ridge alpha.")
    specs = feature_group_specs if feature_group_specs is not None else get_default_alpha_sweep_specs()
    split = split_by_session(rows, held_out_session=held_out_session)
    results: list[AlphaSweepResult] = []
    for name, feature_groups in specs:
        resolved_groups = resolve_feature_groups(feature_groups)
        for ridge_alpha in ridge_alphas:
            metrics = compute_metrics(
                split.train_rows,
                split.test_rows,
                held_out_session=split.held_out_session,
                target_column=target_column,
                ridge_alpha=ridge_alpha,
                feature_groups=resolved_groups,
            )
            results.append(
                AlphaSweepResult(
                    name=name,
                    feature_groups=metrics.feature_groups,
                    feature_count=metrics.feature_count,
                    ridge_alpha=ridge_alpha,
                    train_count=metrics.train_count,
                    test_count=metrics.test_count,
                    dropped_train_count=metrics.dropped_train_count,
                    dropped_test_count=metrics.dropped_test_count,
                    held_out_session=metrics.held_out_session,
                    target_column=metrics.target_column,
                    mae=metrics.mae,
                    rmse=metrics.rmse,
                )
            )
    return results


def run_leave_one_session_out(
    rows: list[dict],
    target_column: str,
    ridge_alpha: float = 1.0,
    feature_groups: list[str] | None = None,
) -> tuple[list[RegressionMetrics], CrossSessionSummary]:
    sessions = list_sessions(rows)
    if len(sessions) < 2:
        raise ValueError("Leave-one-session-out evaluation requires at least two sessions.")
    metrics_by_session: list[RegressionMetrics] = []
    for held_out_session in sessions:
        split = split_by_session(rows, held_out_session=held_out_session)
        metrics_by_session.append(
            compute_metrics(
                split.train_rows,
                split.test_rows,
                held_out_session=split.held_out_session,
                target_column=target_column,
                ridge_alpha=ridge_alpha,
                feature_groups=feature_groups,
            )
        )
    summary = CrossSessionSummary(
        target_column=target_column,
        feature_groups=metrics_by_session[0].feature_groups,
        feature_count=metrics_by_session[0].feature_count,
        ridge_alpha=ridge_alpha,
        session_count=len(metrics_by_session),
        mean_mae=sum(metric.mae for metric in metrics_by_session) / len(metrics_by_session),
        mean_rmse=sum(metric.rmse for metric in metrics_by_session) / len(metrics_by_session),
    )
    return metrics_by_session, summary
