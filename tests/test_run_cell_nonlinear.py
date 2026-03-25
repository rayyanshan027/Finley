from __future__ import annotations

import unittest

from finley.models.run_cell_nonlinear import (
    TreeRegressorConfig,
    build_feature_matrix,
    compute_nonlinear_metrics,
    fit_session_unit_feature_encoder,
    fit_random_forest,
    get_nonlinear_feature_count,
    predict_forest,
    run_leave_one_session_out_nonlinear,
)


def make_row(
    *,
    session: int,
    environment: str,
    mean_speed: float,
    std_speed: float,
    speed_q75: float,
    fast_fraction: float,
    target: float,
) -> dict:
    return {
        "session": session,
        "epoch": 1,
        "task_environment": environment,
        "task_exposure": 1,
        "task_experimentday": session + 4,
        "pos_rows": 10 + session,
        "epoch_duration_sec": 9.0 + 0.1 * session,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "max_speed": mean_speed + 1.5,
        "mean_accel": 0.1 * mean_speed,
        "std_accel": 0.05 * std_speed,
        "mean_abs_accel": 0.1 * mean_speed,
        "max_abs_accel": 0.2 * mean_speed,
        "speed_q25": max(mean_speed - 0.7, 0.0),
        "speed_q50": mean_speed,
        "speed_q75": speed_q75,
        "moving_fraction": min(0.3 + 0.05 * mean_speed, 1.0),
        "fast_fraction": fast_fraction,
        "path_length": 20.0 + mean_speed,
        "step_length_mean": 2.0 + 0.1 * mean_speed,
        "step_length_max": 3.0 + 0.2 * mean_speed,
        "x_range": 10.0 + mean_speed,
        "y_range": 5.0 + 0.5 * mean_speed,
        "rawpos_rows": 12 + session,
        "spike_tetrode_count": 2,
        "spike_cell_count": 4,
        "spike_event_rows_epoch": 12 + int(target * 2),
        "tetrode": 1,
        "cell": 1,
        "depth": 100.0 + session,
        "spikewidth": 8.0 + 0.1 * session,
        "num_spikes": 3 + int(target),
        "log_firing_rate_hz": target,
        "log_num_spikes": target,
    }


class RunCellNonlinearTests(unittest.TestCase):
    def test_build_feature_matrix_uses_selected_groups(self) -> None:
        row = make_row(
            session=3,
            environment="TrackA",
            mean_speed=1.0,
            std_speed=0.2,
            speed_q75=1.5,
            fast_fraction=0.1,
            target=1.0,
        )
        matrix = build_feature_matrix([row], feature_groups=["task_context", "cell_metadata"])
        self.assertEqual(len(matrix), 1)
        self.assertEqual(len(matrix[0]), 7)
        self.assertEqual(get_nonlinear_feature_count(["movement_summaries"]), 19)

    def test_build_feature_matrix_appends_session_unit_identity_one_hot(self) -> None:
        first = make_row(
            session=6,
            environment="TrackA",
            mean_speed=1.0,
            std_speed=0.2,
            speed_q75=1.5,
            fast_fraction=0.1,
            target=1.0,
        )
        second = {
            **make_row(
                session=6,
                environment="TrackA",
                mean_speed=1.1,
                std_speed=0.25,
                speed_q75=1.6,
                fast_fraction=0.1,
                target=1.1,
            ),
            "cell": 2,
        }
        encoder = fit_session_unit_feature_encoder([first, second])
        matrix = build_feature_matrix(
            [first, second],
            feature_groups=["movement_summaries"],
            session_unit_encoder=encoder,
        )
        self.assertEqual(len(matrix[0]), 21)
        self.assertEqual(matrix[0][-2:], [1.0, 0.0])
        self.assertEqual(matrix[1][-2:], [0.0, 1.0])
        self.assertEqual(get_nonlinear_feature_count(["movement_summaries"], encoder), 21)

    def test_build_feature_matrix_leaves_unknown_session_unit_all_zero(self) -> None:
        known = make_row(
            session=6,
            environment="TrackA",
            mean_speed=1.0,
            std_speed=0.2,
            speed_q75=1.5,
            fast_fraction=0.1,
            target=1.0,
        )
        unknown = {
            **make_row(
                session=6,
                environment="TrackB",
                mean_speed=1.2,
                std_speed=0.3,
                speed_q75=1.8,
                fast_fraction=0.2,
                target=1.2,
            ),
            "cell": 3,
        }
        encoder = fit_session_unit_feature_encoder([known])
        matrix = build_feature_matrix(
            [unknown],
            feature_groups=["movement_summaries"],
            session_unit_encoder=encoder,
        )
        self.assertEqual(matrix[0][-1:], [0.0])

    def test_random_forest_fits_simple_signal(self) -> None:
        rows = [
            make_row(
                session=3,
                environment="TrackA" if index % 2 == 0 else "TrackB",
                mean_speed=0.8 + 0.2 * index,
                std_speed=0.1 + 0.03 * index,
                speed_q75=1.0 + 0.25 * index,
                fast_fraction=0.05 * (index % 4),
                target=0.5 if index < 4 else 1.5,
            )
            for index in range(8)
        ]
        x = build_feature_matrix(rows, feature_groups=["movement_summaries"])
        y = [float(row["log_firing_rate_hz"]) for row in rows]
        forest = fit_random_forest(
            x,
            y,
            config=TreeRegressorConfig(
                n_estimators=12,
                max_depth=4,
                min_samples_leaf=1,
                max_features="all",
                random_seed=3,
            ),
        )
        predictions = predict_forest(x, forest)
        mae = sum(abs(prediction - actual) for prediction, actual in zip(predictions, y)) / len(y)
        self.assertLess(mae, 0.3)

    def test_compute_nonlinear_metrics_filters_missing_targets(self) -> None:
        train_rows = [
            make_row(
                session=3,
                environment="TrackA",
                mean_speed=1.0,
                std_speed=0.2,
                speed_q75=1.5,
                fast_fraction=0.1,
                target=1.0,
            ),
            {
                **make_row(
                    session=3,
                    environment="TrackB",
                    mean_speed=1.2,
                    std_speed=0.3,
                    speed_q75=1.8,
                    fast_fraction=0.2,
                    target=1.1,
                ),
                "log_firing_rate_hz": None,
            },
        ]
        test_rows = [
            make_row(
                session=4,
                environment="TrackA",
                mean_speed=1.4,
                std_speed=0.3,
                speed_q75=2.0,
                fast_fraction=0.2,
                target=1.2,
            )
        ]
        metrics = compute_nonlinear_metrics(
            train_rows,
            test_rows,
            held_out_session=4,
            target_column="log_firing_rate_hz",
            feature_groups=["movement_summaries"],
            config=TreeRegressorConfig(
                n_estimators=8,
                max_depth=3,
                min_samples_leaf=1,
                max_features="all",
                random_seed=1,
            ),
        )
        self.assertEqual(metrics.train_count, 1)
        self.assertEqual(metrics.test_count, 1)
        self.assertEqual(metrics.dropped_train_count, 1)
        self.assertEqual(metrics.feature_count, 19)

    def test_compute_nonlinear_metrics_includes_session_unit_features(self) -> None:
        train_rows = [
            make_row(
                session=6,
                environment="TrackA",
                mean_speed=1.0,
                std_speed=0.2,
                speed_q75=1.5,
                fast_fraction=0.1,
                target=1.0,
            ),
            {
                **make_row(
                    session=6,
                    environment="TrackA",
                    mean_speed=1.1,
                    std_speed=0.2,
                    speed_q75=1.6,
                    fast_fraction=0.1,
                    target=1.1,
                ),
                "cell": 2,
            },
            make_row(
                session=5,
                environment="TrackB",
                mean_speed=0.9,
                std_speed=0.2,
                speed_q75=1.4,
                fast_fraction=0.1,
                target=0.8,
            ),
        ]
        test_rows = [
            make_row(
                session=6,
                environment="TrackA",
                mean_speed=1.2,
                std_speed=0.25,
                speed_q75=1.7,
                fast_fraction=0.2,
                target=1.2,
            )
        ]
        encoder = fit_session_unit_feature_encoder(train_rows[:2])
        metrics = compute_nonlinear_metrics(
            train_rows,
            test_rows,
            held_out_session=6,
            target_column="log_firing_rate_hz",
            feature_groups=["movement_summaries"],
            config=TreeRegressorConfig(
                n_estimators=8,
                max_depth=3,
                min_samples_leaf=1,
                max_features="all",
                random_seed=1,
            ),
            session_unit_encoder=encoder,
        )
        self.assertEqual(metrics.feature_count, 21)

    def test_run_leave_one_session_out_nonlinear_returns_each_session(self) -> None:
        rows = []
        for session in [3, 4, 5]:
            for offset in [0.0, 0.3, 0.6]:
                signal = session * 0.2 + offset
                rows.append(
                    make_row(
                        session=session,
                        environment="TrackA" if offset < 0.5 else "TrackB",
                        mean_speed=signal,
                        std_speed=0.2 + 0.05 * offset,
                        speed_q75=signal + 0.8,
                        fast_fraction=0.1 + 0.05 * offset,
                        target=0.4 + signal,
                    )
                )
        metrics_by_session, summary = run_leave_one_session_out_nonlinear(
            rows,
            target_column="log_firing_rate_hz",
            feature_groups=["movement_summaries"],
            config=TreeRegressorConfig(
                n_estimators=10,
                max_depth=4,
                min_samples_leaf=1,
                max_features="sqrt",
                random_seed=2,
            ),
        )
        self.assertEqual([metric.held_out_session for metric in metrics_by_session], [3, 4, 5])
        self.assertEqual(summary.session_count, 3)
        self.assertEqual(summary.feature_count, 19)


if __name__ == "__main__":
    unittest.main()
