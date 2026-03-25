from __future__ import annotations

from dataclasses import dataclass
import math
import random

from finley.models.run_cell_baseline import (
    FEATURE_GROUP_COLUMNS,
    _feature_map,
    filter_rows_by_environment,
    filter_rows_for_target,
    get_available_feature_groups,
    list_sessions,
    load_model_table,
    resolve_feature_groups,
    split_by_session,
)


@dataclass(frozen=True)
class TreeRegressorConfig:
    n_estimators: int = 48
    max_depth: int = 6
    min_samples_leaf: int = 8
    max_features: str | int = "sqrt"
    random_seed: int = 0


@dataclass(frozen=True)
class TreeNode:
    prediction: float
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


@dataclass(frozen=True)
class NonlinearRegressionMetrics:
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
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: str | int
    random_seed: int


@dataclass(frozen=True)
class NonlinearCrossSessionSummary:
    target_column: str
    feature_groups: list[str]
    feature_count: int
    session_count: int
    mean_mae: float
    mean_rmse: float
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: str | int
    random_seed: int


@dataclass(frozen=True)
class SessionUnitFeatureEncoder:
    unit_keys: tuple[tuple[int, int, int], ...]

    @property
    def feature_count(self) -> int:
        return len(self.unit_keys)


def _resolve_feature_names(feature_groups: list[str] | None = None) -> list[str]:
    groups = resolve_feature_groups(feature_groups)
    feature_names: list[str] = []
    for group in groups:
        feature_names.extend(FEATURE_GROUP_COLUMNS[group])
    return feature_names


def get_session_unit_key(row: dict) -> tuple[int, int, int]:
    return int(row["session"]), int(row["tetrode"]), int(row["cell"])


def fit_session_unit_feature_encoder(rows: list[dict]) -> SessionUnitFeatureEncoder:
    unit_keys = tuple(sorted({get_session_unit_key(row) for row in rows}))
    return SessionUnitFeatureEncoder(unit_keys=unit_keys)


def build_feature_matrix(
    rows: list[dict],
    feature_groups: list[str] | None = None,
    session_unit_encoder: SessionUnitFeatureEncoder | None = None,
) -> list[list[float]]:
    feature_names = _resolve_feature_names(feature_groups)
    session_unit_index = (
        {key: index for index, key in enumerate(session_unit_encoder.unit_keys)}
        if session_unit_encoder is not None
        else {}
    )
    matrix: list[list[float]] = []
    for row in rows:
        feature_map = _feature_map(row)
        feature_row = [float(feature_map[name]) for name in feature_names]
        if session_unit_encoder is not None:
            unit_features = [0.0] * session_unit_encoder.feature_count
            unit_index = session_unit_index.get(get_session_unit_key(row))
            if unit_index is not None:
                unit_features[unit_index] = 1.0
            feature_row.extend(unit_features)
        matrix.append(feature_row)
    return matrix


def get_nonlinear_feature_count(
    feature_groups: list[str] | None = None,
    session_unit_encoder: SessionUnitFeatureEncoder | None = None,
) -> int:
    feature_count = len(_resolve_feature_names(feature_groups))
    if session_unit_encoder is not None:
        feature_count += session_unit_encoder.feature_count
    return feature_count


def _variance(y: list[float]) -> float:
    if not y:
        return 0.0
    mean = sum(y) / len(y)
    return sum((value - mean) ** 2 for value in y) / len(y)


def _sum_squared_error(total: float, total_sq: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total_sq - (total * total) / count


def _resolve_max_features(max_features: str | int, total_feature_count: int) -> int:
    if total_feature_count <= 0:
        raise ValueError("Tree model requires at least one feature.")
    if isinstance(max_features, int):
        return max(1, min(max_features, total_feature_count))
    if max_features == "sqrt":
        return max(1, int(math.sqrt(total_feature_count)))
    if max_features == "all":
        return total_feature_count
    raise ValueError(f"Unsupported max_features value: {max_features}")


def _best_split_for_feature(
    x: list[list[float]],
    y: list[float],
    feature_index: int,
    min_samples_leaf: int,
) -> tuple[float, float] | None:
    pairs = sorted((row[feature_index], target) for row, target in zip(x, y))
    if not pairs:
        return None

    total_count = len(pairs)
    total_sum = sum(target for _, target in pairs)
    total_sum_sq = sum(target * target for _, target in pairs)
    left_count = 0
    left_sum = 0.0
    left_sum_sq = 0.0
    best_score = math.inf
    best_threshold = 0.0

    for index in range(total_count - 1):
        value, target = pairs[index]
        left_count += 1
        left_sum += target
        left_sum_sq += target * target
        right_count = total_count - left_count
        next_value = pairs[index + 1][0]
        if value == next_value:
            continue
        if left_count < min_samples_leaf or right_count < min_samples_leaf:
            continue
        right_sum = total_sum - left_sum
        right_sum_sq = total_sum_sq - left_sum_sq
        score = _sum_squared_error(left_sum, left_sum_sq, left_count) + _sum_squared_error(
            right_sum,
            right_sum_sq,
            right_count,
        )
        if score < best_score:
            best_score = score
            best_threshold = (value + next_value) / 2.0

    if best_score == math.inf:
        return None
    return best_score, best_threshold


def _fit_tree(
    x: list[list[float]],
    y: list[float],
    depth: int,
    config: TreeRegressorConfig,
    rng: random.Random,
) -> TreeNode:
    prediction = sum(y) / len(y)
    if (
        depth >= config.max_depth
        or len(y) <= config.min_samples_leaf * 2
        or _variance(y) < 1e-12
    ):
        return TreeNode(prediction=prediction)

    feature_count = len(x[0])
    candidate_count = _resolve_max_features(config.max_features, feature_count)
    feature_indices = list(range(feature_count))
    if candidate_count < feature_count:
        feature_indices = rng.sample(feature_indices, candidate_count)

    best_feature_index = None
    best_threshold = None
    best_score = math.inf
    for feature_index in feature_indices:
        candidate = _best_split_for_feature(x, y, feature_index, config.min_samples_leaf)
        if candidate is None:
            continue
        score, threshold = candidate
        if score < best_score:
            best_score = score
            best_feature_index = feature_index
            best_threshold = threshold

    if best_feature_index is None or best_threshold is None:
        return TreeNode(prediction=prediction)

    left_x: list[list[float]] = []
    left_y: list[float] = []
    right_x: list[list[float]] = []
    right_y: list[float] = []
    for row, target in zip(x, y):
        if row[best_feature_index] <= best_threshold:
            left_x.append(row)
            left_y.append(target)
        else:
            right_x.append(row)
            right_y.append(target)

    if not left_y or not right_y:
        return TreeNode(prediction=prediction)

    left = _fit_tree(left_x, left_y, depth + 1, config, rng)
    right = _fit_tree(right_x, right_y, depth + 1, config, rng)
    return TreeNode(
        prediction=prediction,
        feature_index=best_feature_index,
        threshold=best_threshold,
        left=left,
        right=right,
    )


def fit_random_forest(
    x_train: list[list[float]],
    y_train: list[float],
    config: TreeRegressorConfig | None = None,
) -> list[TreeNode]:
    if not x_train:
        raise ValueError("Cannot fit nonlinear baseline on empty training data.")
    resolved_config = config or TreeRegressorConfig()
    forest: list[TreeNode] = []
    bootstrap_rng = random.Random(resolved_config.random_seed)
    for estimator_index in range(resolved_config.n_estimators):
        sample_x: list[list[float]] = []
        sample_y: list[float] = []
        for _ in range(len(x_train)):
            sampled_index = bootstrap_rng.randrange(len(x_train))
            sample_x.append(x_train[sampled_index])
            sample_y.append(y_train[sampled_index])
        tree_rng = random.Random(resolved_config.random_seed + estimator_index + 1)
        forest.append(_fit_tree(sample_x, sample_y, depth=0, config=resolved_config, rng=tree_rng))
    return forest


def predict_tree(row: list[float], node: TreeNode) -> float:
    current = node
    while not current.is_leaf:
        assert current.feature_index is not None
        assert current.threshold is not None
        assert current.left is not None
        assert current.right is not None
        if row[current.feature_index] <= current.threshold:
            current = current.left
        else:
            current = current.right
    return current.prediction


def predict_forest(x: list[list[float]], forest: list[TreeNode]) -> list[float]:
    if not forest:
        raise ValueError("Forest is empty.")
    predictions: list[float] = []
    for row in x:
        tree_predictions = [predict_tree(row, tree) for tree in forest]
        predictions.append(sum(tree_predictions) / len(tree_predictions))
    return predictions


def compute_nonlinear_metrics(
    train_rows: list[dict],
    test_rows: list[dict],
    held_out_session: int,
    target_column: str,
    feature_groups: list[str] | None = None,
    config: TreeRegressorConfig | None = None,
    session_unit_encoder: SessionUnitFeatureEncoder | None = None,
) -> NonlinearRegressionMetrics:
    resolved_feature_groups = resolve_feature_groups(feature_groups)
    resolved_config = config or TreeRegressorConfig()
    filtered_train_rows, dropped_train_count = filter_rows_for_target(train_rows, target_column)
    filtered_test_rows, dropped_test_count = filter_rows_for_target(test_rows, target_column)
    if not filtered_train_rows or not filtered_test_rows:
        raise ValueError(
            f"No usable rows remain for target {target_column} after filtering missing targets."
        )

    x_train = build_feature_matrix(
        filtered_train_rows,
        feature_groups=resolved_feature_groups,
        session_unit_encoder=session_unit_encoder,
    )
    y_train = [float(row[target_column]) for row in filtered_train_rows]
    x_test = build_feature_matrix(
        filtered_test_rows,
        feature_groups=resolved_feature_groups,
        session_unit_encoder=session_unit_encoder,
    )
    y_test = [float(row[target_column]) for row in filtered_test_rows]

    forest = fit_random_forest(x_train, y_train, config=resolved_config)
    predictions = predict_forest(x_test, forest)
    errors = [prediction - actual for prediction, actual in zip(predictions, y_test)]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return NonlinearRegressionMetrics(
        train_count=len(filtered_train_rows),
        test_count=len(filtered_test_rows),
        dropped_train_count=dropped_train_count,
        dropped_test_count=dropped_test_count,
        feature_groups=resolved_feature_groups,
        feature_count=get_nonlinear_feature_count(
            resolved_feature_groups,
            session_unit_encoder=session_unit_encoder,
        ),
        held_out_session=held_out_session,
        target_column=target_column,
        mae=mae,
        rmse=rmse,
        n_estimators=resolved_config.n_estimators,
        max_depth=resolved_config.max_depth,
        min_samples_leaf=resolved_config.min_samples_leaf,
        max_features=resolved_config.max_features,
        random_seed=resolved_config.random_seed,
    )


def run_leave_one_session_out_nonlinear(
    rows: list[dict],
    target_column: str,
    feature_groups: list[str] | None = None,
    config: TreeRegressorConfig | None = None,
) -> tuple[list[NonlinearRegressionMetrics], NonlinearCrossSessionSummary]:
    sessions = list_sessions(rows)
    if len(sessions) < 2:
        raise ValueError("Leave-one-session-out evaluation requires at least two sessions.")
    resolved_config = config or TreeRegressorConfig()
    metrics_by_session: list[NonlinearRegressionMetrics] = []
    for held_out_session in sessions:
        split = split_by_session(rows, held_out_session=held_out_session)
        metrics_by_session.append(
            compute_nonlinear_metrics(
                split.train_rows,
                split.test_rows,
                held_out_session=split.held_out_session,
                target_column=target_column,
                feature_groups=feature_groups,
                config=resolved_config,
            )
        )
    summary = NonlinearCrossSessionSummary(
        target_column=target_column,
        feature_groups=metrics_by_session[0].feature_groups,
        feature_count=metrics_by_session[0].feature_count,
        session_count=len(metrics_by_session),
        mean_mae=sum(metric.mae for metric in metrics_by_session) / len(metrics_by_session),
        mean_rmse=sum(metric.rmse for metric in metrics_by_session) / len(metrics_by_session),
        n_estimators=resolved_config.n_estimators,
        max_depth=resolved_config.max_depth,
        min_samples_leaf=resolved_config.min_samples_leaf,
        max_features=resolved_config.max_features,
        random_seed=resolved_config.random_seed,
    )
    return metrics_by_session, summary


__all__ = [
    "NonlinearCrossSessionSummary",
    "NonlinearRegressionMetrics",
    "SessionUnitFeatureEncoder",
    "TreeNode",
    "TreeRegressorConfig",
    "build_feature_matrix",
    "compute_nonlinear_metrics",
    "filter_rows_by_environment",
    "fit_random_forest",
    "fit_session_unit_feature_encoder",
    "get_available_feature_groups",
    "get_nonlinear_feature_count",
    "get_session_unit_key",
    "load_model_table",
    "predict_forest",
    "run_leave_one_session_out_nonlinear",
    "split_by_session",
]
