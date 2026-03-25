from __future__ import annotations

from dataclasses import dataclass
import math


PROFILE_NUMERIC_COLUMNS = [
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
    "rawpos_rows",
    "spike_tetrode_count",
    "spike_cell_count",
    "spike_event_rows_epoch",
    "num_spikes",
    "firing_rate_hz",
    "log_firing_rate_hz",
]


@dataclass(frozen=True)
class SessionGroupSummary:
    name: str
    sessions: list[int]
    row_count: int
    mean_metrics: dict[str, float | None]


def _mean(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = _mean(values) or 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


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


def summarize_session_rows(session: int, rows: list[dict]) -> dict:
    environments = sorted({str(row["task_environment"]) for row in rows if row.get("task_environment")})
    counts_by_environment = {
        environment: sum(1 for row in rows if row.get("task_environment") == environment)
        for environment in environments
    }
    output = {
        "session": session,
        "row_count": len(rows),
        "epoch_count": len({int(row["epoch"]) for row in rows}),
        "tracka_fraction": counts_by_environment.get("TrackA", 0) / len(rows) if rows else 0.0,
        "trackb_fraction": counts_by_environment.get("TrackB", 0) / len(rows) if rows else 0.0,
    }
    for column in PROFILE_NUMERIC_COLUMNS:
        values = _numeric_values(rows, column)
        output[f"{column}_mean"] = _mean(values)
        output[f"{column}_std"] = _std(values)
        output[f"{column}_q50"] = _quantile(values, 0.5)
    return output


def summarize_group(name: str, rows: list[dict], sessions: list[int]) -> SessionGroupSummary:
    mean_metrics: dict[str, float | None] = {}
    summary_row = summarize_session_rows(session=-1, rows=rows)
    for key, value in summary_row.items():
        if key == "session":
            continue
        mean_metrics[key] = value
    return SessionGroupSummary(
        name=name,
        sessions=sessions,
        row_count=len(rows),
        mean_metrics=mean_metrics,
    )


def summarize_model_table_by_session(
    rows: list[dict],
    hard_sessions: list[int] | None = None,
    easy_sessions: list[int] | None = None,
) -> dict:
    sessions = sorted({int(row["session"]) for row in rows})
    session_rows = []
    for session in sessions:
        held_rows = [row for row in rows if int(row["session"]) == session]
        session_rows.append(summarize_session_rows(session, held_rows))

    output: dict[str, object] = {"session_profiles": session_rows}
    group_summaries: list[dict] = []
    if hard_sessions:
        hard_rows = [row for row in rows if int(row["session"]) in set(hard_sessions)]
        group_summaries.append(summarize_group("hard_sessions", hard_rows, hard_sessions).__dict__)
    if easy_sessions:
        easy_rows = [row for row in rows if int(row["session"]) in set(easy_sessions)]
        group_summaries.append(summarize_group("easy_sessions", easy_rows, easy_sessions).__dict__)
    if hard_sessions and easy_sessions:
        hard_rows = [row for row in rows if int(row["session"]) in set(hard_sessions)]
        easy_rows = [row for row in rows if int(row["session"]) in set(easy_sessions)]
        comparison: dict[str, float | None] = {}
        hard_summary = summarize_session_rows(session=-1, rows=hard_rows)
        easy_summary = summarize_session_rows(session=-1, rows=easy_rows)
        for key, hard_value in hard_summary.items():
            if key == "session":
                continue
            easy_value = easy_summary.get(key)
            if isinstance(hard_value, (int, float)) and isinstance(easy_value, (int, float)):
                comparison[key] = float(hard_value) - float(easy_value)
            else:
                comparison[key] = None
        output["group_comparison"] = {
            "hard_sessions": hard_sessions,
            "easy_sessions": easy_sessions,
            "hard_minus_easy": comparison,
        }
    if group_summaries:
        output["group_summaries"] = group_summaries
    return output
