from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def summarize_session_data(config: DatasetConfig, animal: str, session: int) -> dict[str, Any]:
    loaded = load_session_files(config, animal, session)
    spikes_day = _extract_session_array(loaded["spikes"], "spikes", session)
    pos_day = _extract_session_array(loaded["pos"], "pos", session)
    task_day = _extract_session_array(loaded["task"], "task", session)
    rawpos_day = _extract_session_array(loaded["rawpos"], "rawpos", session)

    epoch_count = max(len(_iter_flat_items(task_day)), len(_iter_flat_items(pos_day)), len(_iter_flat_items(rawpos_day)))
    epochs: list[dict[str, Any]] = []

    spikes_epochs = _iter_flat_items(spikes_day)
    pos_epochs = _iter_flat_items(pos_day)
    task_epochs = _iter_flat_items(task_day)
    rawpos_epochs = _iter_flat_items(rawpos_day)

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


def build_epoch_rows(config: DatasetConfig, animal: str, session: int) -> list[dict[str, Any]]:
    loaded = load_session_files(config, animal, session)
    spikes_day = _extract_session_array(loaded["spikes"], "spikes", session)
    pos_day = _extract_session_array(loaded["pos"], "pos", session)
    task_day = _extract_session_array(loaded["task"], "task", session)
    rawpos_day = _extract_session_array(loaded["rawpos"], "rawpos", session)

    spikes_epochs = _iter_flat_items(spikes_day)
    pos_epochs = _iter_flat_items(pos_day)
    task_epochs = _iter_flat_items(task_day)
    rawpos_epochs = _iter_flat_items(rawpos_day)
    epoch_count = max(len(task_epochs), len(pos_epochs), len(rawpos_epochs), len(spikes_epochs))

    rows: list[dict[str, Any]] = []
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
                rows_in_cell = _array_row_count(getattr(cell_struct, "data", None))
                if rows_in_cell > 0:
                    cell_count += 1
                    spike_event_rows += rows_in_cell

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
                "rawpos_rows": _array_row_count(getattr(rawpos_struct, "data", None)),
                "rawpos_fields": _to_python_string(getattr(rawpos_struct, "fields", None)),
                "spike_tetrode_count": tetrode_count,
                "spike_cell_count": cell_count,
                "spike_event_rows": spike_event_rows,
            }
        )
    return rows


def build_cell_rows(config: DatasetConfig, animal: str, session: int) -> list[dict[str, Any]]:
    loaded = load_session_files(config, animal, session)
    spikes_day = _extract_session_array(loaded["spikes"], "spikes", session)
    task_day = _extract_session_array(loaded["task"], "task", session)

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
