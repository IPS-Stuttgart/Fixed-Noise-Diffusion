from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .plotting import save_figure
from .summarize_sample_quality import (
    condition_kind,
    condition_pool_size,
    write_csv,
)
from .utils import float_or_nan, format_float

METRIC_COLUMNS = [
    "fid_mean",
    "denoising_gap_mean",
    "low_mid_mean_timestep_gap",
]


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Input spec must be LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Input label is empty in {spec!r}")
    return label, Path(raw_path).expanduser()


def infer_schedule(label: str, condition: str) -> str:
    text = f"{label}_{condition}".lower()
    if "cosine" in text:
        return "cosine"
    if "linear" in text:
        return "linear"
    return "unknown"


def infer_model(label: str, condition: str) -> str:
    text = f"{label}_{condition}".lower()
    if "strong96" in text:
        return "strong96"
    return "base64"


def _pool_size_from_row(row: dict[str, str]) -> int | None:
    condition = row["condition"]
    parsed = condition_pool_size(condition)
    if parsed is not None:
        return parsed

    for key in ("pool_size", "pool_size_sort"):
        raw_value = row.get(key, "")
        if raw_value in ("", "inf", "None", None):
            continue
        value = float(raw_value)
        if math.isfinite(value):
            return int(value)
    return None


def normalize_summary_row(
    row: dict[str, str], label: str, source_path: Path
) -> dict[str, str]:
    condition = row["condition"]
    pool_size = _pool_size_from_row(row)
    normalized = {
        "series": label,
        "schedule": infer_schedule(label, condition),
        "model": infer_model(label, condition),
        "condition": condition,
        "kind": row.get("kind") or condition_kind(condition),
        "pool_size": "" if pool_size is None else str(pool_size),
        "epoch": row.get("epoch", ""),
        "n": row.get("n", ""),
        "source": str(source_path),
    }
    for column in METRIC_COLUMNS:
        value = row.get(column, "")
        if column == "low_mid_mean_timestep_gap" and value == "":
            value = row.get("mean_timestep_gap", "")
        normalized[column] = format_float(float_or_nan(value))
    for column in ("fid_std", "denoising_gap_std"):
        normalized[column] = format_float(float_or_nan(row.get(column, "")))
    return normalized


def read_phase_rows(input_specs: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in input_specs:
        label, path = parse_input_spec(spec)
        resolved = path.resolve()
        with resolved.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(normalize_summary_row(row, label, resolved))
    return sorted(
        rows,
        key=lambda row: (
            row["model"],
            row["schedule"],
            row["series"],
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
        ),
    )


def _metric_value(row: dict[str, str], column: str) -> float:
    return float_or_nan(row.get(column, ""))


def _metric_std(row: dict[str, str], column: str) -> float:
    std_column = {
        "fid_mean": "fid_std",
        "denoising_gap_mean": "denoising_gap_std",
    }.get(column, "")
    return float_or_nan(row.get(std_column, "")) if std_column else math.nan


def _pool_value(row: dict[str, str]) -> int | None:
    return int(row["pool_size"]) if row.get("pool_size") else None


def _series_groups(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["series"], []).append(row)
    return groups


def plot_phase_diagram(rows: list[dict[str, str]], output: Path) -> None:
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    metric_titles = [
        ("fid_mean", "FID-2048"),
        ("denoising_gap_mean", "Final denoising gap"),
        ("low_mid_mean_timestep_gap", "Low/mid timestep gap"),
    ]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, (metric, title) in zip(axes, metric_titles):
        for index, (series, group) in enumerate(sorted(_series_groups(rows).items())):
            color = colors[index % len(colors)]
            fixed = [row for row in group if _pool_value(row) is not None]
            fixed = sorted(fixed, key=lambda row: int(row["pool_size"]))
            if fixed:
                x_values = [_pool_value(row) for row in fixed]
                y_values = [_metric_value(row, metric) for row in fixed]
                y_errors = [_metric_std(row, metric) for row in fixed]
                has_errors = any(not math.isnan(value) for value in y_errors)
                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors if has_errors else None,
                    marker="o",
                    capsize=3 if has_errors else 0,
                    label=series,
                    color=color,
                )
            gaussian = [
                _metric_value(row, metric) for row in group if _pool_value(row) is None
            ]
            if gaussian:
                axis.axhline(
                    gaussian[0],
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xlabel("Pool size M")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[0].legend(frameon=False, fontsize=8)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine and plot WP2 fixed-pool phase-diagram summaries."
    )
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Summary CSV as LABEL=PATH. May be passed more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="wp2_phase_diagram")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if not args.input:
        raise ValueError("At least one --input LABEL=PATH is required")

    output_dir = args.output_dir.expanduser()
    rows = read_phase_rows(args.input)
    write_csv(output_dir / f"{args.prefix}_combined.csv", rows)
    if not args.no_plot:
        plot_phase_diagram(rows, output_dir / f"{args.prefix}.png")


if __name__ == "__main__":
    main()
