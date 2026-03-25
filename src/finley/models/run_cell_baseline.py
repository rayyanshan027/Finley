from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class TrainTestSplit:
    train_rows: list[dict]
    test_rows: list[dict]
    held_out_session: int


@dataclass(frozen=True)
class RegressionMetrics:
    train_count: int
    test_count: int
    held_out_session: int
    target_column: str
    mae: float
    rmse: float


@dataclass(frozen=True)
class FeatureScaler:
    means: list[float]
    scales: list[float]


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
        }
        for row in reader:
            parsed = dict(row)
            for field in int_fields:
                parsed[field] = int(parsed[field])
            for field in float_fields:
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


def _feature_vector(row: dict) -> list[float]:
    environment = row["task_environment"]
    track_a = 1.0 if environment == "TrackA" else 0.0
    track_b = 1.0 if environment == "TrackB" else 0.0
    other_epoch_spikes = max(int(row["spike_event_rows_epoch"]) - int(row["num_spikes"]), 0)
    other_epoch_cells = max(int(row["spike_cell_count"]) - 1, 0)

    return [
        1.0,
        track_a,
        track_b,
        float(row["task_exposure"] or 0.0),
        float(row["task_experimentday"] or 0.0),
        float(row["pos_rows"]),
        float(row["epoch_duration_sec"] or 0.0),
        float(row["mean_speed"] or 0.0),
        float(row["std_speed"] or 0.0),
        float(row["max_speed"] or 0.0),
        float(row["speed_q25"] or 0.0),
        float(row["speed_q50"] or 0.0),
        float(row["speed_q75"] or 0.0),
        float(row["moving_fraction"] or 0.0),
        float(row["fast_fraction"] or 0.0),
        float(row["path_length"] or 0.0),
        float(row["step_length_mean"] or 0.0),
        float(row["step_length_max"] or 0.0),
        float(row["x_range"] or 0.0),
        float(row["y_range"] or 0.0),
        float(row["mean_dir"] or 0.0),
        float(row["dir_std"] or 0.0),
        float(row["rawpos_rows"]),
        float(row["spike_tetrode_count"]),
        float(other_epoch_cells),
        float(other_epoch_spikes),
        float(row["tetrode"]),
        float(row["depth"] or 0.0),
        float(row["spikewidth"] or 0.0),
    ]


def build_design_matrix(rows: list[dict], target_column: str) -> tuple[list[list[float]], list[float]]:
    if not rows:
        raise ValueError("Cannot build design matrix from empty rows.")
    x = [_feature_vector(row) for row in rows]
    y = [float(row[target_column]) for row in rows]
    return x, y


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
) -> RegressionMetrics:
    x_train, y_train = build_design_matrix(train_rows, target_column=target_column)
    x_test, y_test = build_design_matrix(test_rows, target_column=target_column)
    scaler = fit_feature_scaler(x_train)
    x_train_scaled = apply_feature_scaler(x_train, scaler)
    x_test_scaled = apply_feature_scaler(x_test, scaler)
    coefficients = fit_ridge_regression(x_train_scaled, y_train, alpha=ridge_alpha)
    predictions = predict(x_test_scaled, coefficients)
    errors = [prediction - actual for prediction, actual in zip(predictions, y_test)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return RegressionMetrics(
        train_count=len(train_rows),
        test_count=len(test_rows),
        held_out_session=held_out_session,
        target_column=target_column,
        mae=mae,
        rmse=rmse,
    )
