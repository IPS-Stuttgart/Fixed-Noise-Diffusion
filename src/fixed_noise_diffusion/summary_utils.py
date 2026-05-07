from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def format_float(value: float | None, precision: int = 12) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{precision}g}"


def _finite_values(values: Sequence[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def sample_mean(values: Sequence[float]) -> float:
    clean = _finite_values(values)
    return mean(clean) if clean else math.nan


def sample_std(values: Sequence[float]) -> float:
    clean = _finite_values(values)
    return stdev(clean) if len(clean) >= 2 else math.nan


def sample_sem(values: Sequence[float]) -> float:
    clean = _finite_values(values)
    return stdev(clean) / math.sqrt(len(clean)) if len(clean) >= 2 else math.nan


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fieldnames) if fieldnames is not None else list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
