from __future__ import annotations

from dataclasses import is_dataclass
from pathlib import Path
from typing import Any


def load_mat_file(path: str | Path) -> dict[str, Any]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required to read HC-6 .mat files. Install it in your environment, "
            "for example with: python -m pip install scipy"
        ) from exc

    return loadmat(
        Path(path),
        squeeze_me=False,
        struct_as_record=False,
        simplify_cells=False,
    )


def summarize_mat_value(value: Any, max_items: int = 10) -> dict[str, Any]:
    summary: dict[str, Any] = {"python_type": type(value).__name__}

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    size = getattr(value, "size", None)

    if shape is not None:
        summary["shape"] = list(shape)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if size is not None:
        summary["size"] = int(size)

    if isinstance(value, str):
        summary["preview"] = value[:120]
        return summary

    if isinstance(value, bytes):
        summary["preview"] = value[:40].decode("utf-8", errors="replace")
        return summary

    if isinstance(value, (int, float, bool)):
        summary["value"] = value
        return summary

    if isinstance(value, list):
        summary["length"] = len(value)
        if value:
            summary["items"] = [summarize_mat_value(item, max_items=3) for item in value[:max_items]]
        return summary

    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value.keys())[:max_items]
        return summary

    # MATLAB files are commonly loaded as numpy ndarrays of dtype object.
    # Sampling the first few flattened items gives a useful view into nested cells/structs.
    if type(value).__name__ == "ndarray":
        flattened = getattr(value, "flat", None)
        if flattened is not None:
            items = []
            for index, item in enumerate(flattened):
                if index >= max_items:
                    break
                items.append(summarize_mat_value(item, max_items=3))
            if items:
                summary["sample_items"] = items
        return summary

    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        summary["field_names"] = list(field_names)[:max_items]
        field_samples: dict[str, Any] = {}
        for field_name in field_names[:max_items]:
            try:
                field_value = getattr(value, field_name)
            except AttributeError:
                continue
            field_samples[str(field_name)] = summarize_mat_value(field_value, max_items=3)
        if field_samples:
            summary["field_samples"] = field_samples
        return summary

    if is_dataclass(value):
        summary["field_names"] = sorted(value.__dataclass_fields__.keys())[:max_items]
        return summary

    return summary


def summarize_mat_dict(contents: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in sorted(contents.items()):
        if key.startswith("__"):
            continue
        summary[key] = summarize_mat_value(value)
    return summary
