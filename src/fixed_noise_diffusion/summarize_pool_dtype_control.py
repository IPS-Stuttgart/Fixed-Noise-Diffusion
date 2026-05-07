from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import yaml

from .plotting import save_figure


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _noise_cfg(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("noise")
    return value if isinstance(value, dict) else {}


def _data_cfg(config: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    value = config.get("data")
    if isinstance(value, dict):
        return value
    value = metadata.get("data")
    return value if isinstance(value, dict) else {}


def _pool_size(config: dict[str, Any], metadata: dict[str, Any], row: dict[str, Any]) -> int | None:
    meta_noise = metadata.get("noise") if isinstance(metadata.get("noise"), dict) else {}
    for value in (row.get("pool_size"), meta_noise.get("pool_size"), _noise_cfg(config).get("pool_size")):
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def read_eval_rows(root: Path, epoch: int | None = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.jsonl")):
        run_dir = metrics_path.parent
        config = _read_yaml(run_dir / "config.yaml")
        metadata = _read_json(run_dir / "run_metadata.json")
        summary = _read_json(run_dir / "run_summary.json")
        dtype = str(_noise_cfg(config).get("pool_dtype", ""))
        if dtype not in {"float16", "float32"}:
            continue
        eval_rows = [row for row in _read_jsonl(metrics_path) if row.get("type") == "eval"]
        if epoch is None and eval_rows:
            eval_rows = [max(eval_rows, key=lambda row: int(row.get("epoch", -1)))]
        elif epoch is not None:
            eval_rows = [row for row in eval_rows if _as_int(row.get("epoch")) == epoch]
        for row in eval_rows:
            pool_size = _pool_size(config, metadata, row)
            if pool_size is None:
                continue
            rows.append(
                {
                    "run_name": config.get("run_name") or metadata.get("run_name") or summary.get("run_name") or run_dir.name,
                    "dataset": str(_data_cfg(config, metadata).get("dataset", "")).lower(),
                    "seed": config.get("seed", metadata.get("seed", summary.get("seed", ""))),
                    "pool_size": pool_size,
                    "pool_dtype": dtype,
                    "epoch": int(row["epoch"]),
                    "step": int(row.get("step", 0)),
                    "train_den_loss": float(row["train_den_loss"]),
                    "gaussian_den_loss": float(row["gaussian_den_loss"]),
                    "denoising_gap": float(row["denoising_gap"]),
                    "source_metrics": str(metrics_path),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (str(row["dataset"]), int(row["pool_size"]), int(row["epoch"]), str(row["pool_dtype"]))
        grouped[key].append(float(row["denoising_gap"]))

    summary: list[dict[str, Any]] = []
    for (dataset, pool_size, epoch, dtype), gaps in sorted(grouped.items()):
        gap_std = stdev(gaps) if len(gaps) > 1 else 0.0
        summary.append(
            {
                "dataset": dataset,
                "pool_size": pool_size,
                "pool_dtype": dtype,
                "epoch": epoch,
                "n": len(gaps),
                "denoising_gap_mean": mean(gaps),
                "denoising_gap_std": gap_std,
                "denoising_gap_sem": gap_std / math.sqrt(len(gaps)),
                "float32_minus_float16_gap_mean": "",
            }
        )
    by_key = {(row["dataset"], row["pool_size"], row["epoch"], row["pool_dtype"]): row for row in summary}
    for row in summary:
        other_dtype = "float16" if row["pool_dtype"] == "float32" else "float32"
        other = by_key.get((row["dataset"], row["pool_size"], row["epoch"], other_dtype))
        if other is not None:
            float32 = row if row["pool_dtype"] == "float32" else other
            float16 = row if row["pool_dtype"] == "float16" else other
            row["float32_minus_float16_gap_mean"] = float(float32["denoising_gap_mean"]) - float(float16["denoising_gap_mean"])
    return summary


def _fmt(value: Any) -> Any:
    return f"{value:.10g}" if isinstance(value, float) else value


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column, "")) for column in columns})


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for dtype in sorted({row["pool_dtype"] for row in rows}):
        dtype_rows = sorted((row for row in rows if row["pool_dtype"] == dtype), key=lambda row: int(row["pool_size"]))
        ax.errorbar(
            [int(row["pool_size"]) for row in dtype_rows],
            [float(row["denoising_gap_mean"]) for row in dtype_rows],
            yerr=[float(row["denoising_gap_std"]) for row in dtype_rows],
            marker="o",
            capsize=3,
            label=dtype,
        )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Pool size M")
    ax.set_ylabel(r"Denoising gap $\Delta_{\mathrm{denoise}}$")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CIFAR-10 fixed-pool dtype control artifacts.")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="wp2_cifar10_pool_dtype_control_100ep")
    parser.add_argument("--epoch", type=int, default=100, help="Use negative value for each run's final eval row.")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    epoch = None if args.epoch < 0 else args.epoch
    eval_rows = read_eval_rows(args.runs_root.expanduser(), epoch=epoch)
    if not eval_rows:
        raise ValueError(f"No dtype-control eval rows found under {args.runs_root}")
    summary_rows = summarize(eval_rows)
    out = args.output_dir.expanduser()
    write_csv(out / f"{args.prefix}_eval_rows.csv", eval_rows, ["run_name", "dataset", "seed", "pool_size", "pool_dtype", "epoch", "step", "train_den_loss", "gaussian_den_loss", "denoising_gap", "source_metrics"])
    write_csv(out / f"{args.prefix}_gap_summary.csv", summary_rows, ["dataset", "pool_size", "pool_dtype", "epoch", "n", "denoising_gap_mean", "denoising_gap_std", "denoising_gap_sem", "float32_minus_float16_gap_mean"])
    if not args.no_plot:
        plot_summary(out / f"{args.prefix}.png", summary_rows)
    print(f"eval_rows={len(eval_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    print(f"output_dir={out}")


if __name__ == "__main__":
    main()
