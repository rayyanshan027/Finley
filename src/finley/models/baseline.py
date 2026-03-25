from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineResult:
    row_count: int
    feature_columns: list[str]
    mean_features: dict[str, float]
    extension_distribution: dict[str, int]


def run_baseline(records: list[dict], feature_columns: list[str]) -> BaselineResult:
    if not records:
        raise ValueError("Inventory is empty. Build inventory against a populated dataset first.")

    missing = [column for column in feature_columns if column not in records[0]]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    means = {
        column: sum(float(record[column]) for record in records) / len(records)
        for column in feature_columns
    }
    extension_distribution: dict[str, int] = {}
    for record in records:
        extension = str(record["extension"])
        extension_distribution[extension] = extension_distribution.get(extension, 0) + 1

    return BaselineResult(
        row_count=len(records),
        feature_columns=feature_columns,
        mean_features=means,
        extension_distribution=dict(sorted(extension_distribution.items())),
    )
