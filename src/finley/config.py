from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    animals: list[str]
    allowed_extensions: list[str]
    ignore_hidden: bool = True


@dataclass(frozen=True)
class InventoryConfig:
    output_csv: Path
    output_json: Path


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int
    feature_columns: list[str]
    min_rows: int = 10


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    inventory: InventoryConfig
    training: TrainingConfig


def load_config(config_path: str | Path) -> ProjectConfig:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = json.load(handle)

    dataset = raw["dataset"]
    inventory = raw["inventory"]
    training = raw["training"]

    return ProjectConfig(
        dataset=DatasetConfig(
            root=Path(dataset["root"]).expanduser(),
            animals=list(dataset.get("animals", [])),
            allowed_extensions=[ext.lower() for ext in dataset.get("allowed_extensions", [])],
            ignore_hidden=bool(dataset.get("ignore_hidden", True)),
        ),
        inventory=InventoryConfig(
            output_csv=Path(inventory["output_csv"]),
            output_json=Path(inventory["output_json"]),
        ),
        training=TrainingConfig(
            random_seed=int(training.get("random_seed", 7)),
            feature_columns=list(training.get("feature_columns", ["size_bytes", "path_depth"])),
            min_rows=int(training.get("min_rows", 10)),
        ),
    )
