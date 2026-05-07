from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import float_or_nan, format_float, sample_mean, sample_std, write_csv_rows

POOL_RE = re.compile(r"(?:fixed_pool|fixed_pool_whitened)_(?P<size>\d+)(?P<unit>k?)$")


def condition_kind(condition: str) -> str:
    if condition == "gaussian" or condition.endswith("_gaussian"):
        return "gaussian"
    if "whitened" in condition:
        return "whitened"
    return "fixed_pool"


def condition_pool_size(condition: str) -> int | None:
    if condition == "gaussian":
        return None
    match = POOL_RE.search(condition)
    if match is None:
        return None
    size = int(match.group("size"))
    if match.group("unit") == "k":
        size *= 1000
    return size


def find_quality_csvs(paths: list[Path]) -> list[Path]:
    csvs: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved.is_file():
            csvs.append(resolved)
            continue
        candidate = resolved / "sample_quality.csv"
        if candidate.is_file():
            csvs.append(candidate)
            continue
        csvs.extend(sorted(resolved.rglob("sample_quality.csv")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in csvs:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise FileNotFoundError("No sample_quality.csv files found")
    return unique


def read_quality_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in find_quality_csvs(paths):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition = row["condition"]
                row = dict(row)
                row["source_csv"] = str(path)
                row["kind"] = condition_kind(condition)
                pool_size = condition_pool_size(condition)
                row["pool_size"] = "" if pool_size is None else str(pool_size)
                rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["kind"],
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
            int(row.get("seed") or -1),
            int(row.get("epoch") or -1),
        ),
    )


def summarize_quality(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["condition"], row["pool_size"], row["epoch"])].append(
            row
        )

    summary: list[dict[str, str]] = []
    for (kind, condition, pool_size, epoch), group in grouped.items():
        fids = [float_or_nan(row.get("fid")) for row in group]
        kids = [float_or_nan(row.get("kid_mean")) for row in group]
        seconds = [float_or_nan(row.get("seconds")) for row in group]
        summary.append(
            {
                "kind": kind,
                "condition": condition,
                "pool_size": pool_size,
                "epoch": epoch,
                "n": str(len(group)),
                "fid_mean": format_float(sample_mean(fids)),
                "fid_std": format_float(sample_std(fids)),
                "kid_mean_mean": format_float(sample_mean(kids)),
                "kid_mean_std": format_float(sample_std(kids)),
                "seconds_mean": format_float(sample_mean(seconds)),
            }
        )
    return sorted(
        summary,
        key=lambda row: (
            row["kind"],
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
            int(row.get("epoch") or -1),
        ),
    )


def read_gap_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.expanduser().open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition = row["condition"]
                current_epoch = int(row.get("epoch") or 0)
                prior_epoch = int(latest.get(condition, {}).get("epoch") or -1)
                if current_epoch < prior_epoch:
                    continue
                gap_mean = row.get(
                    "denoising_gap_mean", row.get("mean_denoising_gap", "")
                )
                gap_std = row.get("denoising_gap_std", row.get("std_denoising_gap", ""))
                latest[condition] = {
                    "epoch": str(current_epoch),
                    "denoising_gap_mean": gap_mean,
                    "denoising_gap_std": gap_std,
                }
    return latest


def merge_gap_summary(
    quality_summary: list[dict[str, str]], gap_rows: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for row in quality_summary:
        merged_row = dict(row)
        gap = gap_rows.get(row["condition"], {})
        merged_row["denoising_gap_mean"] = gap.get("denoising_gap_mean", "")
        merged_row["denoising_gap_std"] = gap.get("denoising_gap_std", "")
        merged.append(merged_row)
    return merged


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    write_csv_rows(path, rows)


def plot_fid_by_pool(summary: list[dict[str, str]], output: Path) -> None:
    fixed = [
        row
        for row in summary
        if row["kind"] == "fixed_pool" and row["pool_size"] and row.get("epoch")
    ]
    if not fixed:
        return
    final_epoch = max(int(row["epoch"]) for row in fixed)
    fixed = [row for row in fixed if int(row["epoch"]) == final_epoch]
    fixed = sorted(fixed, key=lambda row: int(row["pool_size"]))
    gaussian = [
        row
        for row in summary
        if row["kind"] == "gaussian" and int(row.get("epoch") or -1) == final_epoch
    ]

    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.errorbar(
        [int(row["pool_size"]) for row in fixed],
        [float_or_nan(row.get("fid_mean", "")) for row in fixed],
        yerr=[
            0.0 if math.isnan(float_or_nan(row.get("fid_std", ""))) else float_or_nan(row.get("fid_std", ""))
            for row in fixed
        ],
        marker="o",
        capsize=3,
        label="fixed pool",
    )
    if gaussian:
        fid_mean = float_or_nan(gaussian[0].get("fid_mean", ""))
        fid_std = float_or_nan(gaussian[0].get("fid_std", ""))
        axis.axhline(
            fid_mean,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Gaussian ({fid_mean:.1f})",
        )
        if not math.isnan(fid_std):
            xmin = int(fixed[0]["pool_size"])
            xmax = int(fixed[-1]["pool_size"])
            axis.fill_between(
                [xmin, xmax],
                fid_mean - fid_std,
                fid_mean + fid_std,
                color="black",
                alpha=0.08,
            )
    axis.set_xscale("log")
    axis.set_xlabel("Pool size M")
    axis.set_ylabel("FID")
    axis.set_title(f"Sample quality at epoch {final_epoch}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_fid_vs_gap(summary: list[dict[str, str]], output: Path) -> None:
    rows = [
        row
        for row in summary
        if row.get("denoising_gap_mean", "") != "" and row.get("fid_mean", "") != ""
    ]
    if not rows:
        return
    final_epoch = max(int(row["epoch"]) for row in rows)
    rows = [row for row in rows if int(row["epoch"]) == final_epoch]
    markers = {"gaussian": "s", "fixed_pool": "o", "whitened": "^"}

    fig, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for kind in ["gaussian", "fixed_pool", "whitened"]:
        group = [row for row in rows if row["kind"] == kind]
        if not group:
            continue
        axis.scatter(
            [float_or_nan(row.get("denoising_gap_mean", "")) for row in group],
            [float_or_nan(row.get("fid_mean", "")) for row in group],
            marker=markers[kind],
            label=kind.replace("_", " "),
            s=46,
        )
        for row in group:
            label = row["condition"]
            label = (
                "G"
                if label == "gaussian"
                else label.replace("fixed_pool_whitened_", "w").replace(
                    "fixed_pool_", ""
                )
            )
            axis.annotate(
                label,
                (
                    float_or_nan(row.get("denoising_gap_mean", "")),
                    float_or_nan(row.get("fid_mean", "")),
                ),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axis.set_xlabel("Denoising gap")
    axis.set_ylabel("FID")
    axis.set_title(f"Gap vs sample quality at epoch {final_epoch}")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine and summarize sample quality CSVs from WP2 runs."
    )
    parser.add_argument(
        "--quality",
        action="append",
        type=Path,
        required=True,
        help=(
            "A sample_quality.csv file or a directory containing one or more "
            "such files."
        ),
    )
    parser.add_argument(
        "--gap-summary",
        action="append",
        type=Path,
        default=[],
        help="Optional denoising-gap summary CSV to join by condition.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="sample_quality")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = read_quality_rows(args.quality)
    summary = summarize_quality(rows)
    output_dir = args.output_dir.expanduser()
    write_csv(output_dir / f"{args.prefix}_combined.csv", rows)
    write_csv(output_dir / f"{args.prefix}_summary.csv", summary)

    if args.gap_summary:
        summary = merge_gap_summary(summary, read_gap_rows(args.gap_summary))
        write_csv(output_dir / f"{args.prefix}_summary_with_gap.csv", summary)

    if not args.no_plots:
        plot_fid_by_pool(summary, output_dir / f"{args.prefix}_fid_by_pool_size.png")
        plot_fid_vs_gap(summary, output_dir / f"{args.prefix}_fid_vs_gap.png")


if __name__ == "__main__":
    main()
