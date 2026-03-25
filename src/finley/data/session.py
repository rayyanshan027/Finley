from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

