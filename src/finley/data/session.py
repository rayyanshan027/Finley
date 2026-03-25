from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import math

from finley.config import DatasetConfig
from finley.data.matlab import load_mat_file, summarize_mat_dict


@dataclass(frozen=True)
class HC6SessionPaths:
    animal: str
    session: int
    animal_root: Path
    spikes_path: Path
    pos_path: Path
    task_path: Path
    rawpos_path: Path


def resolve_animal_root(config: DatasetConfig, animal: str) -> Path:
    candidate = config.root / animal
    if candidate.exists():
        return candidate
    if config.root.name.lower() == animal.lower() and config.root.exists():
        return config.root
    raise FileNotFoundError(f"Could not resolve animal root for {animal} under {config.root}")


def build_session_paths(config: DatasetConfig, animal: str, session: int) -> HC6SessionPaths:
    animal_root = resolve_animal_root(config, animal)
    prefix = animal.lower()
    suffix = f"{session:02d}.mat"
    return HC6SessionPaths(
        animal=animal,
        session=session,
        animal_root=animal_root,
        spikes_path=animal_root / f"{prefix}spikes{suffix}",
        pos_path=animal_root / f"{prefix}pos{suffix}",
        task_path=animal_root / f"{prefix}task{suffix}",
        rawpos_path=animal_root / f"{prefix}rawpos{suffix}",
    )


def list_available_sessions(config: DatasetConfig, animal: str) -> list[int]:
    animal_root = resolve_animal_root(config, animal)
    pattern = re.compile(rf"^{re.escape(animal.lower())}spikes(?P<session>\d{{2}})\.mat$")
    sessions: set[int] = set()
    for path in animal_root.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name.lower())
        if match:
            sessions.add(int(match.group("session")))
    return sorted(sessions)


def inspect_session_files(config: DatasetConfig, animal: str, session: int) -> dict[str, dict]:
    paths = build_session_paths(config, animal, session)
    outputs: dict[str, dict] = {}
    for label, path in {
        "spikes": paths.spikes_path,
        "pos": paths.pos_path,
        "task": paths.task_path,
        "rawpos": paths.rawpos_path,
    }.items():
        if not path.exists():
            outputs[label] = {"path": str(path), "exists": False}
            continue

        contents = load_mat_file(path)
        outputs[label] = {
            "path": str(path),
            "exists": True,
            "summary": summarize_mat_dict(contents),
        }
    return outputs


def load_session_files(config: DatasetConfig, animal: str, session: int) -> dict[str, dict]:
    paths = build_session_paths(config, animal, session)
    outputs: dict[str, dict] = {}
    for label, path in {
        "spikes": paths.spikes_path,
        "pos": paths.pos_path,
        "task": paths.task_path,
        "rawpos": paths.rawpos_path,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"Expected {label} file for session {session} at {path}")
        outputs[label] = load_mat_file(path)
    return outputs


def _is_array_like(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "size") and hasattr(value, "flat")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if _is_array_like(value):
        return int(value.size) == 0
    return False


def _scalarize(value: Any) -> Any:
    current = value
    while _is_array_like(current) and int(current.size) == 1:
        if hasattr(current, "item"):
            try:
                current = current.item()
                continue
            except ValueError:
                pass
        current = next(iter(current.flat))
    return current


def _to_python_string(value: Any) -> str | None:
    scalar = _scalarize(value)
    if scalar is None:
        return None
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="replace")
    text = str(scalar).strip()
    return text or None


def _array_row_count(value: Any) -> int:
    scalar = _scalarize(value)
    shape = getattr(scalar, "shape", None)
    if shape is None or len(shape) == 0:
        return 0
    return int(shape[0])


def _iter_flat_items(value: Any) -> list[Any]:
    scalar = _scalarize(value)
    if not _is_array_like(scalar):
        return [] if _is_empty(scalar) else [scalar]
    if int(scalar.size) == 0:
        return []
    return [item for item in scalar.flat]


def _extract_session_array(contents: dict[str, Any], key: str, session: int) -> Any:
    root = contents[key]
    if not _is_array_like(root):
        raise ValueError(f"Expected array-like MATLAB value for {key}")
    if len(root.shape) < 2:
        return root
    if root.shape[1] >= session:
        candidate = root[0, session - 1]
        if not _is_empty(candidate):
            return candidate
    for item in root.flat:
        if not _is_empty(item):
            return item
    return root


def _extract_loaded_session_arrays(loaded: dict[str, dict], session: int) -> dict[str, Any]:
    return {
        "spikes": _extract_session_array(loaded["spikes"], "spikes", session),
        "pos": _extract_session_array(loaded["pos"], "pos", session),
        "task": _extract_session_array(loaded["task"], "task", session),
        "rawpos": _extract_session_array(loaded["rawpos"], "rawpos", session),
    }


def summarize_session_data(config: DatasetConfig, animal: str, session: int) -> dict[str, Any]:
    loaded = load_session_files(config, animal, session)
    arrays = _extract_loaded_session_arrays(loaded, session)
    spikes_day = arrays["spikes"]
    pos_day = arrays["pos"]
    task_day = arrays["task"]
    rawpos_day = arrays["rawpos"]

    spikes_epochs = _iter_flat_items(spikes_day)
    pos_epochs = _iter_flat_items(pos_day)
    task_epochs = _iter_flat_items(task_day)
    rawpos_epochs = _iter_flat_items(rawpos_day)
    epoch_count = max(len(task_epochs), len(pos_epochs), len(rawpos_epochs), len(spikes_epochs))
    epochs: list[dict[str, Any]] = []

    for epoch_index in range(epoch_count):
        task_epoch = task_epochs[epoch_index] if epoch_index < len(task_epochs) else None
        pos_epoch = pos_epochs[epoch_index] if epoch_index < len(pos_epochs) else None
        rawpos_epoch = rawpos_epochs[epoch_index] if epoch_index < len(rawpos_epochs) else None
        spikes_epoch = spikes_epochs[epoch_index] if epoch_index < len(spikes_epochs) else None

        task_struct = _scalarize(task_epoch)
        pos_struct = _scalarize(pos_epoch)
        rawpos_struct = _scalarize(rawpos_epoch)

        tetrode_count = 0
        cell_count = 0
        spike_event_rows = 0
        for tetrode in _iter_flat_items(spikes_epoch):
            cells = _iter_flat_items(tetrode)
            nonempty_cells = [cell for cell in cells if not _is_empty(cell)]
            if nonempty_cells:
                tetrode_count += 1
            for cell in nonempty_cells:
                cell_struct = _scalarize(cell)
                data = getattr(cell_struct, "data", None)
                rows = _array_row_count(data)
                if rows > 0:
                    cell_count += 1
                    spike_event_rows += rows

        epochs.append(
            {
                "epoch": epoch_index + 1,
                "task_type": _to_python_string(getattr(task_struct, "type", None)),
                "task_description": _to_python_string(getattr(task_struct, "description", None)),
                "pos_fields": _to_python_string(getattr(pos_struct, "fields", None)),
                "rawpos_fields": _to_python_string(getattr(rawpos_struct, "fields", None)),
                "rawpos_rows": _array_row_count(getattr(rawpos_struct, "data", None)),
                "spike_tetrode_count": tetrode_count,
                "spike_cell_count": cell_count,
                "spike_event_rows": spike_event_rows,
            }
        )

    return {
        "animal": animal,
        "session": session,
        "epoch_count": epoch_count,
        "epochs": epochs,
    }


def _to_python_number(value: Any) -> int | float | None:
    scalar = _scalarize(value)
    if scalar is None:
        return None
    if hasattr(scalar, "item"):
        try:
            scalar = scalar.item()
        except ValueError:
            pass
    if isinstance(scalar, (int, float)):
        return scalar
    return None


def _normalize_task_type(value: Any) -> str | None:
    text = _to_python_string(value)
    return text.lower() if text else None


def _to_float_list(value: Any) -> list[float]:
    scalar = _scalarize(value)
    if _is_empty(scalar):
        return []
    if _is_array_like(scalar):
        output: list[float] = []
        for item in scalar.flat:
            item_scalar = _scalarize(item)
            number = _to_python_number(item_scalar)
            if number is not None:
                output.append(float(number))
        return output
    number = _to_python_number(scalar)
    return [float(number)] if number is not None else []


def _extract_pos_feature_map(pos_struct: Any) -> dict[str, float | None]:
    empty_result = {
        "epoch_duration_sec": None,
        "mean_speed": None,
        "std_speed": None,
        "max_speed": None,
        "speed_q25": None,
        "speed_q50": None,
        "speed_q75": None,
        "moving_fraction": None,
        "fast_fraction": None,
        "path_length": None,
        "step_length_mean": None,
        "step_length_max": None,
        "x_range": None,
        "y_range": None,
    }
    data = _scalarize(getattr(pos_struct, "data", None))
    if _is_empty(data) or not _is_array_like(data):
        return empty_result

    fields_text = _to_python_string(getattr(pos_struct, "fields", None)) or ""
    field_names = fields_text.split()
    field_index = {field_name: index for index, field_name in enumerate(field_names)}
    shape = getattr(data, "shape", ())
    if len(shape) < 2 or shape[1] == 0:
        return empty_result

    def column_values(column_name: str) -> list[float]:
        index = field_index.get(column_name)
        if index is None or index >= shape[1]:
            return []
        return _to_float_list(data[:, index])

    time_values = column_values("time")
    x_values = column_values("x")
    y_values = column_values("y")
    vel_values = column_values("vel")

    def quantile(values: list[float], q: float) -> float | None:
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

    epoch_duration_sec = None
    if len(time_values) >= 2:
        epoch_duration_sec = float(time_values[-1] - time_values[0])

    mean_speed = sum(vel_values) / len(vel_values) if vel_values else None
    std_speed = None
    if vel_values:
        mean = mean_speed or 0.0
        std_speed = math.sqrt(sum((value - mean) ** 2 for value in vel_values) / len(vel_values))
    max_speed = max(vel_values) if vel_values else None
    speed_q25 = quantile(vel_values, 0.25)
    speed_q50 = quantile(vel_values, 0.50)
    speed_q75 = quantile(vel_values, 0.75)
    moving_fraction = None
    if vel_values:
        moving_count = sum(1 for value in vel_values if value > 0.0)
        moving_fraction = moving_count / len(vel_values)
    fast_fraction = None
    if vel_values:
        fast_count = sum(1 for value in vel_values if value >= 5.0)
        fast_fraction = fast_count / len(vel_values)

    step_lengths: list[float] = []
    if len(x_values) >= 2 and len(y_values) >= 2:
        for index in range(1, min(len(x_values), len(y_values))):
            dx = x_values[index] - x_values[index - 1]
            dy = y_values[index] - y_values[index - 1]
            step_lengths.append(math.sqrt(dx * dx + dy * dy))
    path_length = sum(step_lengths) if step_lengths else None
    step_length_mean = (sum(step_lengths) / len(step_lengths)) if step_lengths else None
    step_length_max = max(step_lengths) if step_lengths else None

    x_range = (max(x_values) - min(x_values)) if x_values else None
    y_range = (max(y_values) - min(y_values)) if y_values else None
    return {
        "epoch_duration_sec": epoch_duration_sec,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "max_speed": max_speed,
        "speed_q25": speed_q25,
        "speed_q50": speed_q50,
        "speed_q75": speed_q75,
        "moving_fraction": moving_fraction,
        "fast_fraction": fast_fraction,
        "path_length": path_length,
        "step_length_mean": step_length_mean,
        "step_length_max": step_length_max,
        "x_range": x_range,
        "y_range": y_range,
    }


def build_epoch_rows(config: DatasetConfig, animal: str, session: int) -> list[dict[str, Any]]:
    loaded = load_session_files(config, animal, session)
    return build_epoch_rows_from_loaded(loaded, animal, session)


def build_epoch_rows_from_loaded(
    loaded: dict[str, dict],
    animal: str,
    session: int,
    cell_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    arrays = _extract_loaded_session_arrays(loaded, session)
    spikes_day = arrays["spikes"]
    pos_day = arrays["pos"]
    task_day = arrays["task"]
    rawpos_day = arrays["rawpos"]

    spikes_epochs = _iter_flat_items(spikes_day)
    pos_epochs = _iter_flat_items(pos_day)
    task_epochs = _iter_flat_items(task_day)
    rawpos_epochs = _iter_flat_items(rawpos_day)
    epoch_count = max(len(task_epochs), len(pos_epochs), len(rawpos_epochs), len(spikes_epochs))

    epoch_spike_stats: dict[int, dict[str, Any]] = {}
    if cell_rows is not None:
        for row in cell_rows:
            epoch = int(row["epoch"])
            stats = epoch_spike_stats.setdefault(
                epoch,
                {"tetrodes": set(), "cell_count": 0, "spike_event_rows": 0},
            )
            stats["tetrodes"].add(int(row["tetrode"]))
            stats["cell_count"] += 1
            stats["spike_event_rows"] += int(row["num_spikes"])

    rows: list[dict[str, Any]] = []
    for epoch_index in range(epoch_count):
        task_epoch = task_epochs[epoch_index] if epoch_index < len(task_epochs) else None
        pos_epoch = pos_epochs[epoch_index] if epoch_index < len(pos_epochs) else None
        rawpos_epoch = rawpos_epochs[epoch_index] if epoch_index < len(rawpos_epochs) else None

        task_struct = _scalarize(task_epoch)
        pos_struct = _scalarize(pos_epoch)
        rawpos_struct = _scalarize(rawpos_epoch)
        pos_features = _extract_pos_feature_map(pos_struct)

        stats = epoch_spike_stats.get(
            epoch_index + 1,
            {"tetrodes": set(), "cell_count": 0, "spike_event_rows": 0},
        )

        rows.append(
            {
                "animal": animal,
                "session": session,
                "epoch": epoch_index + 1,
                "task_type": _normalize_task_type(getattr(task_struct, "type", None)),
                "task_environment": _to_python_string(getattr(task_struct, "environment", None)),
                "task_description": _to_python_string(getattr(task_struct, "description", None)),
                "task_exposure": _to_python_number(getattr(task_struct, "exposure", None)),
                "task_experimentday": _to_python_number(getattr(task_struct, "experimentday", None)),
                "pos_rows": _array_row_count(getattr(pos_struct, "data", None)),
                "pos_fields": _to_python_string(getattr(pos_struct, "fields", None)),
                "epoch_duration_sec": pos_features["epoch_duration_sec"],
                "mean_speed": pos_features["mean_speed"],
                "std_speed": pos_features["std_speed"],
                "max_speed": pos_features["max_speed"],
                "speed_q25": pos_features["speed_q25"],
                "speed_q50": pos_features["speed_q50"],
                "speed_q75": pos_features["speed_q75"],
                "moving_fraction": pos_features["moving_fraction"],
                "fast_fraction": pos_features["fast_fraction"],
                "path_length": pos_features["path_length"],
                "step_length_mean": pos_features["step_length_mean"],
                "step_length_max": pos_features["step_length_max"],
                "x_range": pos_features["x_range"],
                "y_range": pos_features["y_range"],
                "rawpos_rows": _array_row_count(getattr(rawpos_struct, "data", None)),
                "rawpos_fields": _to_python_string(getattr(rawpos_struct, "fields", None)),
                "spike_tetrode_count": len(stats["tetrodes"]),
                "spike_cell_count": int(stats["cell_count"]),
                "spike_event_rows": int(stats["spike_event_rows"]),
            }
        )
    return rows


def build_cell_rows(config: DatasetConfig, animal: str, session: int) -> list[dict[str, Any]]:
    loaded = load_session_files(config, animal, session)
    return build_cell_rows_from_loaded(loaded, animal, session)


def build_cell_rows_from_loaded(loaded: dict[str, dict], animal: str, session: int) -> list[dict[str, Any]]:
    arrays = _extract_loaded_session_arrays(loaded, session)
    spikes_day = arrays["spikes"]
    task_day = arrays["task"]

    spikes_epochs = _iter_flat_items(spikes_day)
    task_epochs = _iter_flat_items(task_day)

    rows: list[dict[str, Any]] = []
    for epoch_index, spikes_epoch in enumerate(spikes_epochs, start=1):
        task_epoch = task_epochs[epoch_index - 1] if epoch_index - 1 < len(task_epochs) else None
        task_struct = _scalarize(task_epoch)
        for tetrode_index, tetrode in enumerate(_iter_flat_items(spikes_epoch), start=1):
            for cell_index, cell in enumerate(_iter_flat_items(tetrode), start=1):
                if _is_empty(cell):
                    continue
                cell_struct = _scalarize(cell)
                data = getattr(cell_struct, "data", None)
                num_spikes = _array_row_count(data)
                if num_spikes == 0:
                    continue

                rows.append(
                    {
                        "animal": animal,
                        "session": session,
                        "epoch": epoch_index,
                        "task_type": _normalize_task_type(getattr(task_struct, "type", None)),
                        "task_environment": _to_python_string(getattr(task_struct, "environment", None)),
                        "tetrode": tetrode_index,
                        "cell": cell_index,
                        "num_spikes": num_spikes,
                        "spike_fields": _to_python_string(getattr(cell_struct, "fields", None)),
                        "spike_description": _to_python_string(getattr(cell_struct, "descript", None)),
                        "spikewidth": _to_python_number(getattr(cell_struct, "spikewidth", None)),
                        "depth": _to_python_number(getattr(cell_struct, "depth", None)),
                    }
                )
    return rows


def build_run_cell_model_rows(epoch_rows: list[dict[str, Any]], cell_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    epoch_lookup = {
        (int(row["session"]), int(row["epoch"])): row
        for row in epoch_rows
        if str(row.get("task_type", "")).lower() == "run"
    }

    rows: list[dict[str, Any]] = []
    for row in cell_rows:
        if str(row.get("task_type", "")).lower() != "run":
            continue
        key = (int(row["session"]), int(row["epoch"]))
        epoch = epoch_lookup.get(key)
        if epoch is None:
            continue

        rows.append(
            {
                "animal": row["animal"],
                "session": int(row["session"]),
                "epoch": int(row["epoch"]),
                "task_environment": row["task_environment"],
                "task_exposure": epoch["task_exposure"],
                "task_experimentday": epoch["task_experimentday"],
                "pos_rows": int(epoch["pos_rows"]),
                "epoch_duration_sec": epoch["epoch_duration_sec"],
                "mean_speed": epoch["mean_speed"],
                "std_speed": epoch["std_speed"],
                "max_speed": epoch["max_speed"],
                "speed_q25": epoch["speed_q25"],
                "speed_q50": epoch["speed_q50"],
                "speed_q75": epoch["speed_q75"],
                "moving_fraction": epoch["moving_fraction"],
                "fast_fraction": epoch["fast_fraction"],
                "path_length": epoch["path_length"],
                "step_length_mean": epoch["step_length_mean"],
                "step_length_max": epoch["step_length_max"],
                "x_range": epoch["x_range"],
                "y_range": epoch["y_range"],
                "rawpos_rows": int(epoch["rawpos_rows"]),
                "spike_tetrode_count": int(epoch["spike_tetrode_count"]),
                "spike_cell_count": int(epoch["spike_cell_count"]),
                "spike_event_rows_epoch": int(epoch["spike_event_rows"]),
                "tetrode": int(row["tetrode"]),
                "cell": int(row["cell"]),
                "depth": row["depth"],
                "spikewidth": row["spikewidth"],
                "num_spikes": int(row["num_spikes"]),
                "firing_rate_hz": None,
                "log_num_spikes": None,
                "log_firing_rate_hz": None,
            }
        )

    for row in rows:
        duration = row["epoch_duration_sec"]
        if duration is not None and duration > 0:
            row["firing_rate_hz"] = row["num_spikes"] / duration
            row["log_firing_rate_hz"] = math.log1p(row["firing_rate_hz"])
        row["log_num_spikes"] = math.log1p(row["num_spikes"])

    return rows
