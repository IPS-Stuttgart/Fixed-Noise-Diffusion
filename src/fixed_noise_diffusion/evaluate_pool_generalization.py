from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .checkpoints import load_checkpoint_model, parse_int_list
from .data import make_dataloaders
from .evaluate import denoising_loss
from .noise import (
    FixedPoolNoiseSampler,
    GaussianNoiseSampler,
    make_noise_sampler,
)
from .summarize_sample_quality import condition_kind, write_csv
from .sweep import add_common_sweep_eval_args, run_identity, select_run_dirs
from .utils import (
    float_or_nan,
    format_float,
    resolve_device,
    sample_mean,
    sample_std,
    seed_everything,
)


def prepare_eval_config(
    config: dict[str, Any],
    batch_size: int,
    batches: int,
    data_dir: str | None,
    num_workers: int,
) -> dict[str, Any]:
    prepared = deepcopy(config)
    data_cfg = prepared["data"]
    data_cfg["download"] = True
    data_cfg["eval_batch_size"] = int(batch_size)
    data_cfg["num_workers"] = int(num_workers)
    if data_dir is not None:
        data_cfg["data_dir"] = data_dir
    requested_images = int(batch_size) * int(batches)
    eval_subset = data_cfg.get("eval_subset_size")
    if eval_subset is not None:
        data_cfg["eval_subset_size"] = max(int(eval_subset), requested_images)
    return prepared


def heldout_pool_config(
    config: dict[str, Any],
    pool_seed: int | None = None,
    pool_seed_offset: int = 1_000_003,
) -> dict[str, Any]:
    heldout = deepcopy(config)
    noise_cfg = heldout["noise"]
    if noise_cfg.get("pool_size") is None:
        raise ValueError("Held-out pool evaluation requires noise.pool_size")
    base_seed = int(noise_cfg["pool_seed"])
    noise_cfg["pool_seed"] = (
        int(pool_seed) if pool_seed is not None else base_seed + int(pool_seed_offset)
    )
    return heldout


def _append_record(csv_path: Path, jsonl_path: Path, record: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _loss_or_blank(value: float | None) -> str:
    return "" if value is None else format_float(value)


def evaluate_run_epoch(
    run_dir: Path,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = resolve_device(args.device)
    condition, run_seed = run_identity(run_dir)
    seed_everything(args.seed + max(run_seed, 0) * 1000 + epoch)

    model, diffusion, config, step = load_checkpoint_model(run_dir, epoch, device)
    config = prepare_eval_config(
        config, args.batch_size, args.batches, args.data_dir, args.num_workers
    )
    loaders = make_dataloaders(config)
    train_sampler = make_noise_sampler(config, device, purpose_seed_offset=0)
    gaussian_sampler = GaussianNoiseSampler(
        image_shape=train_sampler.image_shape,
        device=device,
        seed=int(config["seed"]) + args.gaussian_seed_offset + epoch,
    )
    timestep_seed = int(config["seed"]) + args.timestep_seed_offset + epoch

    train_loss = denoising_loss(
        model=model,
        diffusion=diffusion,
        loader=loaders.val,
        sampler=train_sampler.fork(int(config["seed"]) + 10_000 + epoch),
        device=device,
        batches=args.batches,
        seed=timestep_seed,
    )
    gaussian_loss = denoising_loss(
        model=model,
        diffusion=diffusion,
        loader=loaders.val,
        sampler=gaussian_sampler,
        device=device,
        batches=args.batches,
        seed=timestep_seed,
    )

    heldout_loss = None
    heldout_seed: int | str = ""
    if isinstance(train_sampler, FixedPoolNoiseSampler):
        heldout_cfg = heldout_pool_config(
            config,
            pool_seed=args.heldout_pool_seed,
            pool_seed_offset=args.heldout_pool_seed_offset,
        )
        heldout_seed = int(heldout_cfg["noise"]["pool_seed"])
        heldout_sampler = make_noise_sampler(heldout_cfg, device)
        heldout_loss = denoising_loss(
            model=model,
            diffusion=diffusion,
            loader=loaders.val,
            sampler=heldout_sampler.fork(int(config["seed"]) + 10_000 + epoch),
            device=device,
            batches=args.batches,
            seed=timestep_seed,
        )

    info = train_sampler.info
    return {
        "run_name": run_dir.name,
        "condition": condition,
        "kind": condition_kind(condition),
        "pool_size": "" if info.pool_size is None else info.pool_size,
        "seed": run_seed,
        "epoch": epoch,
        "step": step,
        "batches": int(args.batches),
        "batch_size": int(args.batch_size),
        "train_pool_seed": config["noise"].get("pool_seed", ""),
        "heldout_pool_seed": heldout_seed,
        "noise_mode": info.mode,
        "train_noise_loss": format_float(train_loss),
        "heldout_pool_loss": _loss_or_blank(heldout_loss),
        "fresh_gaussian_loss": format_float(gaussian_loss),
        "heldout_pool_gap": _loss_or_blank(
            None if heldout_loss is None else heldout_loss - train_loss
        ),
        "fresh_gaussian_gap": format_float(gaussian_loss - train_loss),
        "gaussian_minus_heldout_gap": _loss_or_blank(
            None if heldout_loss is None else gaussian_loss - heldout_loss
        ),
        "source_run_dir": str(run_dir),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["kind"]),
            str(row["condition"]),
            str(row.get("pool_size", "")),
            str(row["epoch"]),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, str]] = []
    for (kind, condition, pool_size, epoch), group in grouped.items():
        item = {
            "kind": kind,
            "condition": condition,
            "pool_size": pool_size,
            "epoch": epoch,
            "n": str(len(group)),
        }
        for column in (
            "train_noise_loss",
            "heldout_pool_loss",
            "fresh_gaussian_loss",
            "heldout_pool_gap",
            "fresh_gaussian_gap",
            "gaussian_minus_heldout_gap",
        ):
            values = [float_or_nan(row.get(column, "")) for row in group]
            item[f"{column}_mean"] = format_float(sample_mean(values))
            item[f"{column}_std"] = format_float(sample_std(values))
        summary.append(item)

    return sorted(
        summary,
        key=lambda row: (
            row["kind"],
            int(row["pool_size"]) if row["pool_size"] else 10**18,
            row["condition"],
            int(row["epoch"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-pool versus held-out-pool denoising losses."
    )
    add_common_sweep_eval_args(parser, default_epochs="100")
    parser.add_argument("--heldout-pool-seed", type=int, default=None)
    parser.add_argument("--heldout-pool-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--gaussian-seed-offset", type=int, default=20_000)
    parser.add_argument("--timestep-seed-offset", type=int, default=30_000)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    csv_path = output_dir / "pool_generalization.csv"
    jsonl_path = output_dir / "pool_generalization.jsonl"
    epochs = parse_int_list(args.epochs)
    rows: list[dict[str, Any]] = []

    for run_dir in select_run_dirs(args.sweep_dir, args.run):
        for epoch in epochs:
            record = evaluate_run_epoch(run_dir, epoch, args)
            _append_record(csv_path, jsonl_path, record)
            rows.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    write_csv(output_dir / "pool_generalization_summary.csv", summarize_rows(rows))


if __name__ == "__main__":
    main()
