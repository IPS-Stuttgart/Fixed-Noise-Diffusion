"""Frozen protocol, provenance, and preregistered decisions for the study."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import socket
import statistics
import subprocess
from importlib.metadata import version
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from .config import DEFAULT_CONFIG
from .persistence_noise import CONDITIONS


SCHEMA_VERSION = 1
CURVE_EPOCHS = [1, 5, 10, 11, 20, 21, 50, 100]
TIMESTEPS = [0, 25, 50, 100, 200, 400, 600, 800, 999]
T_CRITICAL_DF2 = 4.302652729911275


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_hash() -> str:
    root = Path(__file__).parent
    return digest({str(path.relative_to(root)): file_hash(path) for path in sorted(root.rglob("*.py"))})


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(encoded):
            raise ValueError(f"Immutable artifact differs: {path}") from None


def protocol(smoke: bool = False) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "smoke": smoke,
        "epochs": 10 if smoke else 100, "train_images": 256 if smoke else 50_000,
        "test_images": 64 if smoke else 10_000, "development_images": 32 if smoke else 2048,
        "batch_size": 128, "pool_size": 32 if smoke else 1000,
        "image_shape": [3, 32, 32], "partition_seed": 20261003,
        "curve_epochs": [1, 5, 10] if smoke else CURVE_EPOCHS,
        "timestep_epochs": [10] if smoke else [10, 50, 100], "timesteps": TIMESTEPS,
        "boundary_epochs": [] if smoke else [10, 20], "endpoint_draws": 4,
        "specialization_margin": 0.05, "ordering_margin": 0.005, "mc_multiplier": 3,
        "max_training_runs": 18, "pilot_seed": 0, "confirmation_seeds": [1, 2, 3],
        "conditions": list(CONDITIONS), "selection": "largest_absolute_contrast_then_interval_10",
        "fid": {"fid_feature": 2048, "sample_count": 50_000, "requested_real_count": 50_000,
                "real_count": 50_000, "fake_count": 50_000, "real_split": "train", "sample_steps": 50, "sampler": "ddim"},
    }


def partitions(spec: dict[str, Any]) -> tuple[list[int], list[int]]:
    permutation = torch.randperm(spec["test_images"], generator=torch.Generator().manual_seed(spec["partition_seed"]))
    return permutation[:spec["development_images"]].tolist(), permutation[spec["development_images"]:].tolist()


def frozen_config(spec: dict[str, Any], condition: str, seed: int, data_dir: str, device: str) -> dict[str, Any]:
    if condition not in CONDITIONS or seed not in [0, 1, 2, 3]:
        raise ValueError("Condition or replicate outside the registered study")
    config = deepcopy(DEFAULT_CONFIG)
    config.update(seed=seed, device=device, run_name=f"persistence_{condition}_seed{seed}")
    config["data"].update(
        dataset="fake" if spec["smoke"] else "cifar10", data_dir=data_dir, download=False,
        batch_size=spec["batch_size"], eval_batch_size=128, num_workers=0,
        eval_subset_size=None, fake_train_size=spec["train_images"], fake_val_size=spec["test_images"],
        channels=spec["image_shape"][0], image_size=spec["image_shape"][1],
    )
    config["diffusion"].update(beta_schedule="cosine", num_timesteps=1000)
    config["training"].update(epochs=spec["epochs"], amp=False, lr=2e-4, weight_decay=0.0,
                              checkpoint_epochs=spec["curve_epochs"], grad_accum_steps=1)
    config["evaluation"].update(sample_count=0, fid_feature=2048)
    config["noise"].update(mode="gaussian" if condition == "gaussian" else "fixed_pool",
                           pool_dtype="float32", pool_size=spec["pool_size"], pool_seed=4242 + 1000 * seed,
                           schedule=condition)
    config["persistence"] = {"protocol": spec, "protocol_sha256": digest(spec), "condition": condition}
    return config


def environment(device: torch.device) -> dict[str, Any]:
    import torchvision
    import numpy
    return {
        "host": socket.gethostname(), "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": str(numpy.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "source_sha256": source_hash(),
        "metric_packages": {name: version(name) for name in ("pillow", "matplotlib", "pyyaml")},
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0]
                  if device.type == "cuda" else None,
    }


def validate_endpoint(record: dict[str, Any], spec: dict[str, Any], condition: str, seed: int) -> None:
    expected = {"condition": condition, "seed": seed, "epoch": spec["epochs"], "protocol_sha256": digest(spec),
                "step": spec["epochs"] * (spec["train_images"] // spec["batch_size"]),
                "partition": "development" if seed == 0 else "confirmation"}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Invalid endpoint {condition}/{seed}: {key}")
    if len(record["fresh_losses"]) != 4 or len(record["current_losses"]) != 4:
        raise ValueError("Endpoint requires four evaluation draws per law")
    if not all(math.isfinite(float(x)) for law in ("fresh_losses", "current_losses") for x in record[law]):
        raise ValueError("Nonfinite endpoint")


def validate_replicate(records: dict[str, dict[str, Any]], spec: dict[str, Any], seed: int, conditions: list[str]) -> None:
    if set(records) != set(conditions):
        raise ValueError("Missing or unexpected conditions")
    identities = []
    for condition in conditions:
        validate_endpoint(records[condition], spec, condition, seed)
        identities.append(records[condition]["environment"])
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Paired conditions used different hardware, software, or source")
    if len({record["initial_model_sha256"] for record in records.values()}) != 1:
        raise ValueError("Paired conditions started from different model weights")
    for condition in conditions:
        if condition.startswith("refresh_"):
            refresh, union = records[condition]["noise"], records[condition.replace("refresh_", "union_")]["noise"]
            if refresh["block_sha256"] != union["block_sha256"] or refresh["exposure_sha256"] != union["exposure_sha256"]:
                raise ValueError("Refresh and union endpoints do not match templates/exposure counts")


def paired_contrast(refresh: dict[str, Any], union: dict[str, Any]) -> dict[str, float]:
    differences = [a - b for a, b in zip(refresh["fresh_losses"], union["fresh_losses"], strict=True)]
    return {"difference": statistics.mean(differences), "mc_se": statistics.stdev(differences) / math.sqrt(len(differences))}


def pilot_decision(records: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    validate_replicate(records, spec, 0, list(CONDITIONS))
    persistent, gaussian = records["persistent"], records["gaussian"]
    gap = statistics.mean(persistent["fresh_losses"]) - statistics.mean(persistent["current_losses"])
    excess = statistics.mean(persistent["fresh_losses"]) - statistics.mean(gaussian["fresh_losses"])
    reproduced = gap >= spec["specialization_margin"] and excess >= spec["specialization_margin"]
    contrasts = {str(interval): paired_contrast(records[f"refresh_{interval}"], records[f"union_{interval}"]) for interval in (10, 1)}
    eligible = [interval for interval in (10, 1) if reproduced and
                abs(contrasts[str(interval)]["difference"]) > max(spec["ordering_margin"], spec["mc_multiplier"] * contrasts[str(interval)]["mc_se"])]
    selected = max(eligible, key=lambda interval: (abs(contrasts[str(interval)]["difference"]), interval)) if eligible else None
    return {"passed": selected is not None, "selected_interval": selected,
            "selected_direction": None if selected is None else (1 if contrasts[str(selected)]["difference"] > 0 else -1),
            "persistent_gap": gap, "persistent_excess": excess, "contrasts": contrasts,
            "protocol_sha256": digest(spec), "pilot_sha256": digest(records)}


def confirmation_decision(replicates: dict[int, dict[str, Any]], spec: dict[str, Any], pilot: dict[str, Any]) -> dict[str, Any]:
    if not pilot["passed"] or set(replicates) != {1, 2, 3}:
        raise ValueError("Confirmation requires a passing pilot and three fresh replicates")
    interval = pilot["selected_interval"]
    conditions = ["gaussian", "persistent", f"refresh_{interval}", f"union_{interval}"]
    effects = []
    for seed in (1, 2, 3):
        records = replicates[seed]
        validate_replicate(records, spec, seed, conditions)
        effects.append(paired_contrast(records[f"refresh_{interval}"], records[f"union_{interval}"])["difference"])
    mean = statistics.mean(effects)
    half_width = T_CRITICAL_DF2 * statistics.stdev(effects) / math.sqrt(3)
    interval95 = [mean - half_width, mean + half_width]
    same_direction = all(effect * pilot["selected_direction"] > 0 for effect in effects)
    passed = same_direction and abs(mean) >= spec["ordering_margin"] and (interval95[0] > 0 or interval95[1] < 0)
    return {"passed": passed, "effects": effects, "mean": mean, "ci95": interval95,
            "same_direction": same_direction, "pilot_sha256": digest(pilot), "replicates_sha256": digest(replicates),
            "interpretation": "ordering_effect_supported_in_this_setup" if passed else "inconclusive_not_equivalence"}


def validate_quality(records: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    if not records:
        raise ValueError("No sample-quality records")
    for record in records:
        for key, value in spec["fid"].items():
            if str(record.get(key)) != str(value):
                raise ValueError(f"Incompatible FID protocol: {key}={record.get(key)!r}, expected {value!r}")
        if not math.isfinite(float(record["fid"])):
            raise ValueError("Nonfinite FID")
