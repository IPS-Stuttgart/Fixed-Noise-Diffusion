"""Isolated, recoverable training for the registered persistence intervention."""
from __future__ import annotations

import json
import math
import os
import random
import resource
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import save_config
from .data import make_dataloaders
from .diffusion import GaussianDiffusion
from .evaluate import denoising_loss_from_timesteps
from .model import build_model
from .persistence_noise import ExposurePlan, ScheduledNoiseSampler, tensor_hash
from .persistence_protocol import atomic_json, digest, environment, file_hash, immutable_json, partitions
from .utils import seed_everything, write_csv_rows


def capture_rng() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), numpy_state[2], numpy_state[3], numpy_state[4]],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))


@contextmanager
def isolated_evaluation(model):
    rng, training = capture_rng(), model.training
    try:
        model.eval()
        yield
    finally:
        model.train(training)
        restore_rng(rng)


def evaluation_loader(dataset, indices: list[int], batch_size: int) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0,
                      generator=torch.Generator().manual_seed(610_000), pin_memory=torch.cuda.is_available())


def evaluate_snapshot(model, diffusion, dataset, indices, sampler, config, *, epoch: int, event: str,
                      partition: str, draws: int = 1, timestep: int | None = None, block: int | None = None) -> list[dict[str, Any]]:
    device = sampler.device
    loader = evaluation_loader(dataset, indices, int(config["data"]["eval_batch_size"]))
    active = sampler.active_block if block is None else block
    roles = ["fresh", "current"]
    if sampler.bank is not None:
        roles.append("unseen")
    if sampler.plan.condition.startswith("refresh_") and active > 0:
        roles.append("retired")
    records = []
    with isolated_evaluation(model):
        for role_index, role in enumerate(roles):
            base = sampler.evaluation_sampler(role, 0, block=active)
            for draw in range(draws):
                noise_seed = 700_000 + int(config["seed"]) * 100_000 + role_index * 10_000 + epoch * 10 + draw
                evaluation_noise = base.fork(noise_seed)
                generator = torch.Generator(device=device).manual_seed(900_000 + epoch * 10 + draw)

                def timesteps(batch_size: int) -> torch.Tensor:
                    if timestep is not None:
                        return torch.full((batch_size,), timestep, dtype=torch.long, device=device)
                    return torch.randint(diffusion.num_timesteps, (batch_size,), generator=generator, device=device)

                loss, count = denoising_loss_from_timesteps(
                    model, diffusion, loader, evaluation_noise, device, len(loader), timesteps
                )
                if not math.isfinite(loss) or count != len(indices):
                    raise ValueError("Invalid or incomplete evaluation")
                records.append({
                    "condition": sampler.plan.condition, "seed": config["seed"], "epoch": epoch,
                    "event": event, "partition": partition, "role": role, "draw": draw,
                    "timestep": timestep, "loss": loss, "images": count, "active_block": active,
                    "noise_seed": noise_seed, "image_indices_sha256": digest(indices),
                    "protocol_sha256": config["persistence"]["protocol_sha256"],
                })
    return records


def save_recovery(run_dir: Path, payload: dict[str, Any], retain: bool) -> None:
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "recovery.tmp"
    torch.save(payload, temporary)
    sha = file_hash(temporary)
    temporary.replace(directory / "recovery.pt")
    atomic_json(directory / "recovery.sha256.json", {"sha256": sha, "epoch": payload["epoch"]})
    if retain:
        destination = directory / f"epoch_{payload['epoch']:04d}.pt"
        if destination.exists():
            if file_hash(destination) != sha:
                raise ValueError("Refusing to replace an existing checkpoint")
        else:
            os.link(directory / "recovery.pt", destination)
        immutable_json(destination.with_suffix(".sha256.json"), {"sha256": sha})


def load_recovery(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    path = run_dir / "checkpoints" / "recovery.pt"
    receipt = json.loads(path.with_suffix(".sha256.json").read_text())
    if file_hash(path) != receipt["sha256"]:
        raise ValueError("Corrupt recovery checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("recovery_version") != 1 or state["manifest_sha256"] != digest(manifest):
        raise ValueError("Legacy, foreign, or incompatible recovery checkpoint")
    if state["epoch"] != receipt["epoch"]:
        raise ValueError("Recovery receipt does not match checkpoint")
    return state


def train_run(config: dict[str, Any], run_dir: Path, *, resume: bool = False,
              stop_after_epoch: int | None = None) -> dict[str, Any] | None:
    spec = config["persistence"]["protocol"]
    condition = config["persistence"]["condition"]
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is forbidden")
    if (run_dir / "failure.json").exists():
        raise RuntimeError("Run has a recorded failure; investigate it before explicit recovery")
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    bundle = make_dataloaders(config)
    train_dataset, test_dataset = bundle.train.dataset, bundle.val.dataset
    if len(train_dataset) != spec["train_images"] or len(test_dataset) != spec["test_images"]:
        raise ValueError("Dataset sizes do not match the frozen study")
    development, confirmation = partitions(spec)
    batch_size = spec["batch_size"]
    steps_per_epoch = len(train_dataset) // batch_size
    if not steps_per_epoch:
        raise ValueError("No complete training batches")
    plan = ExposurePlan(condition, config["seed"], spec["epochs"], steps_per_epoch * batch_size,
                        spec["pool_size"], tuple(spec["image_shape"]))
    sampler = ScheduledNoiseSampler(plan, device)
    model = build_model(config).to(device)
    diffusion = GaussianDiffusion.from_config(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=0.0)
    timestep_generator = torch.Generator(device=device).manual_seed(600_000 + config["seed"])
    manifest = {
        "config": config, "environment": environment(device), "initial_noise": sampler.fingerprint(),
        "development_indices_sha256": digest(development), "confirmation_indices_sha256": digest(confirmation),
        "initial_model_sha256": digest({name: tensor_hash(value) for name, value in model.state_dict().items()}),
    }
    immutable_json(run_dir / "manifest.json", manifest)
    if not resume:
        save_config(config, run_dir / "config.yaml")
    rows: list[dict[str, Any]] = []
    epoch_stats: list[dict[str, Any]] = []
    completed_epoch, previous_seconds = 0, 0.0
    if resume:
        state = load_recovery(run_dir, manifest)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["noise_state"])
        timestep_generator.set_state(state["timestep_rng"])
        rows, epoch_stats = state["evaluation_rows"], state["epoch_stats"]
        completed_epoch, previous_seconds = state["epoch"], state["seconds"]
        if state["step"] != completed_epoch * steps_per_epoch:
            raise ValueError("Recovery optimizer-step budget mismatch")
        restore_rng(state["rng"])
    for epoch in range(completed_epoch + 1, spec["epochs"] + 1):
        if shutil.disk_usage(run_dir).free < 4 * 1024**3:
            raise OSError("Less than 4 GiB free; refusing to risk an incomplete recovery checkpoint")
        sampler.begin_epoch(epoch)
        model.train()
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(100_000 + config["seed"] * 1000 + epoch))
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device)
            timesteps = torch.randint(diffusion.num_timesteps, (len(images),), generator=timestep_generator, device=device)
            noise = sampler.sample(len(images))
            noisy = diffusion.q_sample(images, timesteps, noise)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(noisy, timesteps), noise)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach())
            if (batch_index + 1) % 100 == 0:
                print(json.dumps({"condition": condition, "seed": config["seed"], "epoch": epoch,
                                  "step": (epoch - 1) * steps_per_epoch + batch_index + 1,
                                  "train_loss": float(loss.detach())}), flush=True)
        train_seconds = time.perf_counter() - epoch_start
        epoch_stats.append({"epoch": epoch, "loss": loss_sum / len(loader), "training_seconds": train_seconds})
        if epoch in spec["curve_epochs"]:
            rows.extend(evaluate_snapshot(model, diffusion, test_dataset, development, sampler, config,
                                         epoch=epoch, event="curve", partition="development"))
        if epoch in spec["timestep_epochs"]:
            for timestep in spec["timesteps"]:
                rows.extend(evaluate_snapshot(model, diffusion, test_dataset, development, sampler, config,
                                             epoch=epoch, event="timestep", partition="development", timestep=timestep))
        if epoch in spec["boundary_epochs"] and condition.startswith("refresh_") and epoch % plan.interval == 0 and epoch < plan.epochs:
            for event, active in (("pre_refresh", sampler.active_block), ("post_refresh", sampler.active_block + 1)):
                rows.extend(evaluate_snapshot(model, diffusion, test_dataset, development, sampler, config,
                                             epoch=epoch, event=event, partition="development", block=active))
        if epoch == spec["epochs"]:
            final_indices = development if config["seed"] == 0 else confirmation
            rows.extend(evaluate_snapshot(model, diffusion, test_dataset, final_indices, sampler, config,
                                         epoch=epoch, event="endpoint", partition="development" if config["seed"] == 0 else "confirmation",
                                         draws=spec["endpoint_draws"]))
        for value in model.state_dict().values():
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite model state")
        payload = {
            "recovery_version": 1, "manifest_sha256": digest(manifest), "epoch": epoch, "step": epoch * steps_per_epoch,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
            "train_noise_pool_fingerprint": sampler.fingerprint(), "noise_state": sampler.state_dict(),
            "timestep_rng": timestep_generator.get_state(), "rng": capture_rng(),
            "evaluation_rows": rows, "epoch_stats": epoch_stats, "seconds": previous_seconds + time.perf_counter() - started,
        }
        save_recovery(run_dir, payload, retain=epoch in spec["curve_epochs"])
        write_csv_rows(run_dir / "evaluation.csv", rows)
        atomic_json(run_dir / "status.json", {"state": "training" if epoch < spec["epochs"] else "trained",
                                              "epoch": epoch, "step": epoch * steps_per_epoch,
                                              "training_seconds_last_epoch": train_seconds, "seconds": payload["seconds"]})
        print(json.dumps({"condition": condition, "epoch_complete": epoch, "training_seconds": train_seconds}), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and epoch < spec["epochs"]:
            return None
    # Rebuild from checkpoint rows even if a crash happened after checkpointing
    # the final epoch but before exporting the CSV.
    write_csv_rows(run_dir / "evaluation.csv", rows)
    endpoint = [row for row in rows if row["event"] == "endpoint"]
    result = {
        "condition": condition, "seed": config["seed"], "epoch": spec["epochs"], "step": spec["epochs"] * steps_per_epoch,
        "partition": "development" if config["seed"] == 0 else "confirmation", "protocol_sha256": digest(spec),
        "environment": manifest["environment"], "manifest_sha256": digest(manifest), "noise": sampler.fingerprint(),
        "initial_model_sha256": manifest["initial_model_sha256"],
        "fresh_losses": [row["loss"] for row in endpoint if row["role"] == "fresh"],
        "current_losses": [row["loss"] for row in endpoint if row["role"] == "current"],
        "seconds": previous_seconds + time.perf_counter() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_host_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "evaluation_sha256": file_hash(run_dir / "evaluation.csv"), "epoch_stats": epoch_stats,
    }
    immutable_json(run_dir / "endpoint.json", result)
    atomic_json(run_dir / "status.json", {"state": "complete", "epoch": spec["epochs"], "step": result["step"]})
    return result
