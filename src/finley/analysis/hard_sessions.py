from __future__ import annotations

from dataclasses import dataclass
import math


ROW_METRIC_COLUMNS = [
    "log_firing_rate_hz",
    "firing_rate_hz",
    "num_spikes",
    "mean_speed",
    "speed_q75",
    "mean_abs_accel",
    "path_length",
    "spike_cell_count",
    "spike_event_rows_epoch",
    "other_epoch_spikes",
]

EPOCH_METRIC_COLUMNS = [
    "epoch_duration_sec",
    "mean_speed",
    "speed_q75",
    "mean_abs_accel",
    "path_length",
    "spike_cell_count",
    "spike_event_rows_epoch",
]

CELL_METRIC_COLUMNS = [
    "log_firing_rate_hz",
    "firing_rate_hz",
    "num_spikes",
    "depth",
    "spikewidth",
]


@dataclass(frozen=True)
class SummarySlice:
    level: str
    label: str
    row_count: int
    epoch_count: int
    metrics: dict[str, float | int | None]


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_values(rows: list[dict], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if value is None:
            continue
        values.append(float(value))
    return values


def _summarize_metrics(rows: list[dict], metric_columns: list[str]) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}
    for column in metric_columns:
        values = _numeric_values(rows, column)
        metrics[f"{column}_mean"] = sum(values) / len(values) if values else None
        metrics[f"{column}_q10"] = _quantile(values, 0.10)
        metrics[f"{column}_q50"] = _quantile(values, 0.50)
        metrics[f"{column}_q90"] = _quantile(values, 0.90)
    return metrics


def build_epoch_rows(rows: list[dict]) -> list[dict]:
    seen: set[tuple[int, int]] = set()
    epoch_rows: list[dict] = []
    for row in rows:
        key = (int(row["session"]), int(row["epoch"]))
        if key in seen:
            continue
        seen.add(key)
        epoch_rows.append(
            {
                "session": int(row["session"]),
                "epoch": int(row["epoch"]),
                "task_environment": row.get("task_environment"),
                "epoch_duration_sec": row.get("epoch_duration_sec"),
                "mean_speed": row.get("mean_speed"),
                "speed_q75": row.get("speed_q75"),
                "mean_abs_accel": row.get("mean_abs_accel"),
                "path_length": row.get("path_length"),
                "spike_cell_count": row.get("spike_cell_count"),
                "spike_event_rows_epoch": row.get("spike_event_rows_epoch"),
            }
        )
    return epoch_rows


def add_population_context(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        enriched = dict(row)
        enriched["other_epoch_spikes"] = max(
            int(enriched["spike_event_rows_epoch"]) - int(enriched["num_spikes"]),
            0,
        )
        output.append(enriched)
    return output


def summarize_slice(level: str, label: str, rows: list[dict], metric_columns: list[str]) -> SummarySlice:
    epoch_count = len({(int(row["session"]), int(row["epoch"])) for row in rows})
    return SummarySlice(
        level=level,
        label=label,
        row_count=len(rows),
        epoch_count=epoch_count,
        metrics=_summarize_metrics(rows, metric_columns),
    )


def _slice_to_row(summary: SummarySlice) -> dict[str, float | int | None | str]:
    return {
        "level": summary.level,
        "label": summary.label,
        "row_count": summary.row_count,
        "epoch_count": summary.epoch_count,
        **summary.metrics,
    }


def summarize_sessions(rows: list[dict]) -> list[dict[str, float | int | None | str]]:
    sessions = sorted({int(row["session"]) for row in rows})
    output = []
    for session in sessions:
        session_rows = [row for row in rows if int(row["session"]) == session]
        output.append(
            _slice_to_row(
                summarize_slice("session", f"session_{session}", session_rows, ROW_METRIC_COLUMNS)
            )
        )
    return output


def summarize_session_tracks(rows: list[dict]) -> list[dict[str, float | int | None | str]]:
    sessions = sorted({int(row["session"]) for row in rows})
    output = []
    for session in sessions:
        for environment in ["TrackA", "TrackB"]:
            track_rows = [
                row
                for row in rows
                if int(row["session"]) == session and row.get("task_environment") == environment
            ]
            if not track_rows:
                continue
            output.append(
                {
                    **_slice_to_row(
                        summarize_slice(
                            "session_track",
                            f"session_{session}_{environment}",
                            track_rows,
                            ROW_METRIC_COLUMNS,
                        )
                    ),
                    "task_environment": environment,
                    "session": session,
                }
            )
    return output


def summarize_groups(
    rows: list[dict],
    epoch_rows: list[dict],
    hard_sessions: list[int],
    easy_sessions: list[int],
) -> list[dict[str, float | int | None | str]]:
    output: list[dict[str, float | int | None | str]] = []
    specs = [("hard_rows", hard_sessions, rows, ROW_METRIC_COLUMNS), ("easy_rows", easy_sessions, rows, ROW_METRIC_COLUMNS)]
    specs.extend(
        [
            ("hard_epochs", hard_sessions, epoch_rows, EPOCH_METRIC_COLUMNS),
            ("easy_epochs", easy_sessions, epoch_rows, EPOCH_METRIC_COLUMNS),
        ]
    )
    for label, sessions, source_rows, metrics in specs:
        if not sessions:
            continue
        subset = [row for row in source_rows if int(row["session"]) in set(sessions)]
        output.append(_slice_to_row(summarize_slice("group", label, subset, metrics)))
    return output


def build_group_delta(
    rows: list[dict],
    epoch_rows: list[dict],
    hard_sessions: list[int],
    easy_sessions: list[int],
) -> dict[str, dict[str, float | None]]:
    hard_rows = [row for row in rows if int(row["session"]) in set(hard_sessions)]
    easy_rows = [row for row in rows if int(row["session"]) in set(easy_sessions)]
    hard_epochs = [row for row in epoch_rows if int(row["session"]) in set(hard_sessions)]
    easy_epochs = [row for row in epoch_rows if int(row["session"]) in set(easy_sessions)]

    def metric_delta(
        left_rows: list[dict],
        right_rows: list[dict],
        metric_columns: list[str],
    ) -> dict[str, float | None]:
        left = _summarize_metrics(left_rows, metric_columns)
        right = _summarize_metrics(right_rows, metric_columns)
        output: dict[str, float | None] = {}
        for key, left_value in left.items():
            right_value = right.get(key)
            if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                output[key] = float(left_value) - float(right_value)
            else:
                output[key] = None
        return output

    return {
        "row_metrics_hard_minus_easy": metric_delta(hard_rows, easy_rows, ROW_METRIC_COLUMNS),
        "epoch_metrics_hard_minus_easy": metric_delta(hard_epochs, easy_epochs, EPOCH_METRIC_COLUMNS),
    }


def _top_rows_by_metric(
    rows: list[dict],
    metric: str,
    top_n: int,
    base_fields: list[str],
) -> list[dict[str, float | int | str | None]]:
    filtered = [row for row in rows if row.get(metric) is not None]
    ordered = sorted(filtered, key=lambda row: float(row[metric]), reverse=True)[:top_n]
    output: list[dict[str, float | int | str | None]] = []
    for rank, row in enumerate(ordered, start=1):
        record: dict[str, float | int | str | None] = {
            "metric": metric,
            "rank": rank,
            "metric_value": float(row[metric]),
        }
        for field in base_fields:
            record[field] = row.get(field)
        output.append(record)
    return output


def build_outlier_rows(
    rows: list[dict],
    epoch_rows: list[dict],
    hard_sessions: list[int],
    top_n: int = 5,
) -> list[dict[str, float | int | str | None]]:
    output: list[dict[str, float | int | str | None]] = []
    target_rows = [row for row in rows if int(row["session"]) in set(hard_sessions)]
    target_epoch_rows = [row for row in epoch_rows if int(row["session"]) in set(hard_sessions)]

    for session in hard_sessions:
        session_rows = [row for row in target_rows if int(row["session"]) == session]
        session_epochs = [row for row in target_epoch_rows if int(row["session"]) == session]

        for metric in ["log_firing_rate_hz", "firing_rate_hz", "num_spikes"]:
            for record in _top_rows_by_metric(
                session_rows,
                metric,
                top_n,
                ["session", "epoch", "task_environment", "tetrode", "cell"],
            ):
                record["level"] = "cell"
                output.append(record)

        for metric in ["mean_speed", "mean_abs_accel", "spike_cell_count", "spike_event_rows_epoch"]:
            for record in _top_rows_by_metric(
                session_epochs,
                metric,
                top_n,
                ["session", "epoch", "task_environment"],
            ):
                record["level"] = "epoch"
                output.append(record)
    return output


def build_hard_session_diagnostics(
    rows: list[dict],
    hard_sessions: list[int],
    easy_sessions: list[int],
    top_n: int = 5,
) -> dict[str, object]:
    enriched_rows = add_population_context(rows)
    epoch_rows = build_epoch_rows(enriched_rows)
    return {
        "row_count": len(enriched_rows),
        "epoch_row_count": len(epoch_rows),
        "hard_sessions": hard_sessions,
        "easy_sessions": easy_sessions,
        "session_summaries": summarize_sessions(enriched_rows),
        "session_track_summaries": summarize_session_tracks(enriched_rows),
        "group_summaries": summarize_groups(enriched_rows, epoch_rows, hard_sessions, easy_sessions),
        "group_deltas": build_group_delta(enriched_rows, epoch_rows, hard_sessions, easy_sessions),
        "outlier_rows": build_outlier_rows(enriched_rows, epoch_rows, hard_sessions, top_n=top_n),
    }


__all__ = [
    "build_epoch_rows",
    "build_hard_session_diagnostics",
    "build_outlier_rows",
    "summarize_groups",
    "summarize_session_tracks",
    "summarize_sessions",
]
