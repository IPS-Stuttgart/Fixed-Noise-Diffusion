from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_FIELDS = [
    "type",
    "epoch",
    "step",
    "split",
    "loss",
    "train_noise_loss",
    "gaussian_den_loss",
    "train_den_loss",
    "heldout_pool_den_loss",
    "denoising_gap",
    "denoising_eval_timestep_seed",
    "heldout_pool_gap",
    "gaussian_minus_heldout_gap",
    "fid",
    "kid_mean",
    "kid_std",
    "lr",
    "seconds",
    "noise_mode",
    "pool_size",
    "pool_memory_mb",
    "heldout_pool_seed",
    "config_hash",
    "git_commit",
    "samples_path",
]


class MetricLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.jsonl_path = run_dir / "metrics.jsonl"
        self.csv_path = run_dir / "metrics.csv"
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
                )
                writer.writeheader()

    def log(self, record: dict[str, Any]) -> None:
        clean = {
            key: (float(value) if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, sort_keys=True) + "\n")
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=CSV_FIELDS, extrasaction="ignore"
            )
            writer.writerow(clean)
