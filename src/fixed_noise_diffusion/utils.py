from __future__ import annotations

import csv
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def make_run_dir(output_dir: str | Path, run_name: str) -> Path:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "samples").mkdir(exist_ok=True)
    return run_dir


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def generator_for(device: torch.device | str, seed: int) -> torch.Generator:
    device = torch.device(device)
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def format_float(value: float | None, precision: int = 12) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{precision}g}"


def sample_mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return mean(clean) if clean else math.nan


def sample_std(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return stdev(clean) if len(clean) >= 2 else math.nan


def sample_sem(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if len(clean) < 2:
        return math.nan
    return stdev(clean) / math.sqrt(len(clean))


def write_csv_rows(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if fieldnames is None else list(fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def working_directory(path: str | Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
