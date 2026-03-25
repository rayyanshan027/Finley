from __future__ import annotations

import math


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Quantile requires at least one value.")
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


def clip_rows_for_target(
    rows: list[dict],
    target_column: str,
    upper_quantile: float,
) -> tuple[list[dict], float]:
    if not 0.0 < upper_quantile <= 1.0:
        raise ValueError("upper_quantile must be in (0, 1].")
    target_values = [float(row[target_column]) for row in rows if row.get(target_column) is not None]
    if not target_values:
        raise ValueError(f"No usable values found for target column {target_column}.")
    clip_value = quantile(target_values, upper_quantile)
    clipped_rows: list[dict] = []
    for row in rows:
        cloned = dict(row)
        value = cloned.get(target_column)
        if value is not None:
            cloned[target_column] = min(float(value), clip_value)
        clipped_rows.append(cloned)
    return clipped_rows, clip_value


__all__ = ["clip_rows_for_target", "quantile"]
