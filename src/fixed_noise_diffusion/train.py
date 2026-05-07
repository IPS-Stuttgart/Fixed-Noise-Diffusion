from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import add_config_args, load_config, save_config
from .data import make_dataloaders
from .diffusion import GaussianDiffusion
from .evaluate import denoising_loss, first_real_batch, optional_fid_kid, sample_grid
from .integrity import build_run_metadata, build_run_summary, write_json
from .logging_utils import MetricLogger
from .model import build_model
from .noise import FixedPoolNoiseSampler, GaussianNoiseSampler, make_noise_sampler
from .utils import (
    Timer,
    count_parameters,
    generator_for,
    make_run_dir,
    resolve_device,
    seed_everything,
)


def _should_checkpoint(epoch: int, training_cfg: dict[str, Any]) -> bool:
    checkpoint_epochs = training_cfg.get("checkpoint_epochs") or []
    return int(epoch) in {int(value) for value in checkpoint_epochs}


def _save_checkpoint(
    run_dir: Path,
    epoch: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt")


def make_evaluation_samplers(
    train_noise_sampler,
    seed: int,
    epoch: int,
    device: torch.device,
):
    train_eval_sampler = train_noise_sampler.fork(seed + 10_000 + epoch)
    gaussian_eval_sampler = GaussianNoiseSampler(
        image_shape=train_noise_sampler.image_shape,
        device=device,
        seed=seed + 20_000 + epoch,
    )
    return train_eval_sampler, gaussian_eval_sampler


def make_heldout_pool_sampler(
    config: dict[str, Any],
    train_noise_sampler,
    device: torch.device,
):
    eval_cfg = config["evaluation"]
    if not bool(eval_cfg.get("enable_heldout_pool", False)):
        return None
    if not isinstance(train_noise_sampler, FixedPoolNoiseSampler):
        return None

    noise_cfg = config["noise"]
    heldout_pool_seed = eval_cfg.get("heldout_pool_seed")
    if heldout_pool_seed is None:
        heldout_pool_seed = int(noise_cfg["pool_seed"]) + int(
            eval_cfg.get("heldout_pool_seed_offset", 1_000_003)
        )

    return FixedPoolNoiseSampler(
        image_shape=train_noise_sampler.image_shape,
        device=device,
        pool_size=train_noise_sampler.pool_size,
        pool_seed=int(heldout_pool_seed),
        index_seed=int(config["seed"]) + 70_000,
        dtype=str(noise_cfg.get("pool_dtype", "float16")),
        chunk_size=int(noise_cfg.get("pool_chunk_size", 8192)),
        whiten=train_noise_sampler.whiten,
    )


def evaluate_checkpoint(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loaders,
    train_noise_sampler,
    heldout_noise_sampler,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
    logger: MetricLogger,
    epoch: int,
    step: int,
    timer: Timer,
) -> dict[str, Any]:
    eval_cfg = config["evaluation"]
    seed = int(config["seed"])
    train_eval_sampler, gaussian_eval_sampler = make_evaluation_samplers(
        train_noise_sampler, seed, epoch, device
    )
    # Pair the denoising-law comparison over the same validation batches and
    # timestep stream. The noise sampler is the only intended difference between
    # train-law, fresh-Gaussian, and held-out-pool evaluation.
    timestep_seed = seed + 30_000 + epoch

    train_den_loss = denoising_loss(
        model,
        diffusion,
        loaders.val,
        train_eval_sampler,
        device,
        int(eval_cfg["denoising_batches"]),
        timestep_seed,
    )
    gaussian_den_loss = denoising_loss(
        model,
        diffusion,
        loaders.val,
        gaussian_eval_sampler,
        device,
        int(eval_cfg["denoising_batches"]),
        timestep_seed,
    )
    heldout_pool_den_loss = None
    heldout_pool_seed = None
    if heldout_noise_sampler is not None:
        heldout_eval_sampler = heldout_noise_sampler.fork(seed + 10_000 + epoch)
        heldout_pool_seed = heldout_noise_sampler.pool_seed
        heldout_pool_den_loss = denoising_loss(
            model,
            diffusion,
            loaders.val,
            heldout_eval_sampler,
            device,
            int(eval_cfg["denoising_batches"]),
            timestep_seed,
        )
    samples_path = run_dir / "samples" / f"epoch_{epoch:04d}.png"
    samples = sample_grid(
        model,
        diffusion,
        config,
        device,
        samples_path,
        seed + 50_000 + epoch,
    )
    metrics = {"fid": None, "kid_mean": None, "kid_std": None}
    if bool(eval_cfg.get("enable_metrics", False)) and samples.numel() > 0:
        real = first_real_batch(loaders.val, device, samples.shape[0])
        metrics = optional_fid_kid(real, samples, device)

    info = train_noise_sampler.info
    record = {
        "type": "eval",
        "epoch": epoch,
        "step": step,
        "split": "val",
        "train_den_loss": train_den_loss,
        "gaussian_den_loss": gaussian_den_loss,
        "denoising_gap": gaussian_den_loss - train_den_loss,
        "denoising_eval_timestep_seed": timestep_seed,
        "heldout_pool_den_loss": heldout_pool_den_loss,
        "heldout_pool_gap": (
            None
            if heldout_pool_den_loss is None
            else heldout_pool_den_loss - train_den_loss
        ),
        "gaussian_minus_heldout_gap": (
            None
            if heldout_pool_den_loss is None
            else gaussian_den_loss - heldout_pool_den_loss
        ),
        "noise_mode": info.mode,
        "pool_size": info.pool_size,
        "pool_memory_mb": round(info.pool_memory_mb, 3),
        "heldout_pool_seed": heldout_pool_seed,
        "samples_path": str(samples_path),
        "seconds": round(timer.elapsed(), 3),
        **metrics,
    }
    logger.log(record)
    return record


def train(config: dict[str, Any]) -> Path:
    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["device"]))
    run_dir = make_run_dir(config["output_dir"], config["run_name"])
    save_config(config, run_dir / "config.yaml")
    logger = MetricLogger(run_dir)
    timer = Timer()

    loaders = make_dataloaders(config)
    model = build_model(config).to(device)
    diffusion = GaussianDiffusion.from_config(config, device)
    train_noise_sampler = make_noise_sampler(config, device, purpose_seed_offset=0)
    heldout_noise_sampler = make_heldout_pool_sampler(
        config, train_noise_sampler, device
    )
    metadata = build_run_metadata(config, run_dir, device, train_noise_sampler.info)
    write_json(run_dir / "run_metadata.json", metadata)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    train_timestep_generator = generator_for(device, int(config["seed"]) + 60_000)

    logger.log(
        {
            "type": "run_start",
            "epoch": 0,
            "step": 0,
            "noise_mode": train_noise_sampler.info.mode,
            "pool_size": train_noise_sampler.info.pool_size,
            "pool_memory_mb": round(train_noise_sampler.info.pool_memory_mb, 3),
            "config_hash": metadata["config_hash"],
            "git_commit": metadata["git"]["commit"],
            "seconds": 0.0,
            "loss": None,
            "lr": float(config["training"]["lr"]),
        }
    )
    print(
        f"Run {config['run_name']} on {device}; "
        f"{count_parameters(model):,} trainable params; "
        f"noise={train_noise_sampler.info.mode}; "
        f"pool_memory={train_noise_sampler.info.pool_memory_mb:.1f} MB"
    )

    global_step = 0
    epoch = 0
    last_eval_record = None
    max_train_steps = config["training"].get("max_train_steps")
    grad_accum_steps = int(config["training"].get("grad_accum_steps", 1))
    log_interval = int(config["training"].get("log_interval_steps", 100))

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        progress = tqdm(loaders.train, desc=f"epoch {epoch}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, _) in enumerate(progress, start=1):
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            timesteps = torch.randint(
                0,
                diffusion.num_timesteps,
                (batch_size,),
                device=device,
                generator=train_timestep_generator,
                dtype=torch.long,
            )
            noise = train_noise_sampler.sample(batch_size)
            noisy = diffusion.q_sample(images, timesteps, noise)
            with autocast(enabled=amp_enabled):
                pred_noise = model(noisy, timesteps)
                loss = F.mse_loss(pred_noise, noise, reduction="mean")
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

            if batch_index % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_interval == 0 or global_step == 1:
                    lr = optimizer.param_groups[0]["lr"]
                    logger.log(
                        {
                            "type": "train_step",
                            "epoch": epoch,
                            "step": global_step,
                            "split": "train",
                            "loss": float(loss.item()),
                            "lr": float(lr),
                            "noise_mode": train_noise_sampler.info.mode,
                            "pool_size": train_noise_sampler.info.pool_size,
                            "pool_memory_mb": round(
                                train_noise_sampler.info.pool_memory_mb, 3
                            ),
                            "seconds": round(timer.elapsed(), 3),
                        }
                    )
                    progress.set_postfix(loss=f"{loss.item():.4f}")

            if max_train_steps is not None and global_step >= int(max_train_steps):
                break

        if _should_checkpoint(epoch, config["training"]):
            last_eval_record = evaluate_checkpoint(
                model,
                diffusion,
                loaders,
                train_noise_sampler,
                heldout_noise_sampler,
                config,
                device,
                run_dir,
                logger,
                epoch,
                global_step,
                timer,
            )
            if bool(config["training"].get("save_checkpoint", True)):
                _save_checkpoint(run_dir, epoch, global_step, model, optimizer, config)

        if max_train_steps is not None and global_step >= int(max_train_steps):
            break

    elapsed = round(timer.elapsed(), 3)
    logger.log(
        {
            "type": "run_end",
            "epoch": epoch,
            "step": global_step,
            "seconds": elapsed,
            "noise_mode": train_noise_sampler.info.mode,
            "pool_size": train_noise_sampler.info.pool_size,
            "pool_memory_mb": round(train_noise_sampler.info.pool_memory_mb, 3),
        }
    )
    summary = build_run_summary(
        config=config,
        run_dir=run_dir,
        metadata=metadata,
        final_epoch=epoch,
        final_step=global_step,
        seconds=elapsed,
        noise_info=train_noise_sampler.info,
        last_eval=last_eval_record,
    )
    write_json(run_dir / "run_summary.json", summary)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 fixed-noise DDPM run."
    )
    add_config_args(parser)
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    train(config)


if __name__ == "__main__":
    main()
