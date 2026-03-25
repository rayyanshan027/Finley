from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

from finley.config import DatasetConfig


@dataclass(frozen=True)
class HC6FileRecord:
    animal: str
    relative_path: str
    extension: str
    size_bytes: int
    path_depth: int
    modality: str
    session: int | None
    top_level_dir: str


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _infer_metadata(animal: str, relative_path: Path) -> tuple[str, int | None, str]:
    normalized_path = relative_path
    if relative_path.parts and relative_path.parts[0].lower() == animal.lower():
        normalized_path = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path(".")

    top_level_dir = normalized_path.parts[0] if len(normalized_path.parts) > 1 else "."
    stem = normalized_path.stem.lower()

    if top_level_dir.lower() == "eeg":
        eeg_pattern = re.compile(rf"^{re.escape(animal.lower())}eeg(?P<session>\d+)(?:-\d+-\d+)?$")
        eeg_match = eeg_pattern.match(stem)
        if eeg_match:
            return "eeg", int(eeg_match.group("session")), top_level_dir

        session_match = re.search(r"(\d+)$", stem)
        session = int(session_match.group(1)) if session_match else None
        return "eeg", session, top_level_dir

    pattern = re.compile(
        rf"^{re.escape(animal.lower())}(?P<kind>rawpos|pos|spikes|task)(?P<session>\d+)$"
    )
    match = pattern.match(stem)
    if match:
        return match.group("kind"), int(match.group("session")), top_level_dir

    if stem == f"{animal.lower()}cellinfo":
        return "cellinfo", None, top_level_dir
    if stem == f"{animal.lower()}tetinfo":
        return "tetinfo", None, top_level_dir

    return "unknown", None, top_level_dir


def _iter_roots(config: DatasetConfig) -> list[tuple[str, Path]]:
    root = config.root
    if config.animals:
        resolved_roots: list[tuple[str, Path]] = []
        for animal in config.animals:
            candidate = root / animal
            if candidate.exists():
                resolved_roots.append((animal, candidate))
            elif root.name == animal and root.exists():
                resolved_roots.append((animal, root))
            else:
                resolved_roots.append((animal, candidate))
        return resolved_roots

    # Support scanning a single animal directory directly, not only the extracted parent.
    if any(path.is_file() for path in root.iterdir()):
        return [(root.name, root)]

    animal_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    return [(path.name, path) for path in animal_dirs]


def scan_dataset(config: DatasetConfig) -> list[dict]:
    records: list[HC6FileRecord] = []

    for animal, animal_root in _iter_roots(config):
        if not animal_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {animal_root}")

        for path in sorted(animal_root.rglob("*")):
            if not path.is_file():
                continue
            if config.ignore_hidden and _is_hidden(path.relative_to(config.root)):
                continue

            extension = path.suffix.lower()
            if config.allowed_extensions and extension not in config.allowed_extensions:
                continue

            relative_path = path.relative_to(config.root)
            modality, session, top_level_dir = _infer_metadata(animal, relative_path)
            records.append(
                HC6FileRecord(
                    animal=animal,
                    relative_path=str(relative_path),
                    extension=extension or "<none>",
                    size_bytes=path.stat().st_size,
                    path_depth=len(relative_path.parts),
                    modality=modality,
                    session=session,
                    top_level_dir=top_level_dir,
                )
            )

    return [record.__dict__ for record in records]


def summarize_inventory(records: list[dict]) -> dict:
    if not records:
        return {
            "row_count": 0,
            "animals": [],
            "extension_counts": {},
            "total_size_bytes": 0,
        }

    extension_counts = Counter(record["extension"] for record in records)
    modality_counts = Counter(record["modality"] for record in records)
    sessions = sorted(
        {int(record["session"]) for record in records if record["session"] is not None}
    )
    return {
        "row_count": len(records),
        "animals": sorted({record["animal"] for record in records}),
        "extension_counts": dict(sorted(extension_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "sessions": sessions,
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
    }
