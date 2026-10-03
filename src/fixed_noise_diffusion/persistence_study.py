"""Bounded pilot/confirmation runner; see docs/persistence-study.md."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .persistence_noise import CONDITIONS
from .persistence_protocol import (
    atomic_json, confirmation_decision, digest, file_hash, frozen_config, immutable_json,
    partitions, pilot_decision, protocol, source_hash, validate_endpoint, validate_quality,
)
from .persistence_train import train_run
from .utils import write_csv_rows


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another worker owns {path}") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def initialize(root: Path, *, smoke: bool = False, smoke_receipt: Path | None = None) -> None:
    spec = protocol(smoke)
    receipt = None
    if not smoke:
        if smoke_receipt is None:
            raise ValueError("A passed GPU smoke receipt is required before registering the pilot")
        receipt = read_json(smoke_receipt)
        if not receipt["passed"] or receipt["source_sha256"] != source_hash() or not receipt["gpu"]:
            raise ValueError("Smoke receipt is failed, CPU-only, or from different source")
    development, confirmation = partitions(spec)
    manifest = {
        "study": "noise-persistence-v1", "protocol": spec, "protocol_sha256": digest(spec),
        "source_sha256": source_hash(), "development_indices_sha256": digest(development),
        "confirmation_indices_sha256": digest(confirmation), "smoke_receipt": receipt,
    }
    immutable_json(root / "manifest.json", manifest)


def study_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    if manifest["protocol"] != protocol(manifest["protocol"]["smoke"]):
        raise ValueError("Study protocol was modified")
    if manifest["protocol_sha256"] != digest(manifest["protocol"]) or manifest["source_sha256"] != source_hash():
        raise ValueError("Protocol or source hash changed after registration")
    return manifest


def run_path(root: Path, condition: str, seed: int) -> Path:
    return root / "runs" / f"persistence_{condition}_seed{seed}"


def load_endpoint(root: Path, condition: str, seed: int) -> dict[str, Any]:
    study = study_manifest(root)
    directory = run_path(root, condition, seed)
    result = read_json(directory / "endpoint.json")
    receipt = read_json(directory / "endpoint.sha256.json")
    if receipt["sha256"] != file_hash(directory / "endpoint.json"):
        raise ValueError("Endpoint checksum mismatch")
    manifest = read_json(directory / "manifest.json")
    if result["manifest_sha256"] != digest(manifest) or result["evaluation_sha256"] != file_hash(directory / "evaluation.csv"):
        raise ValueError("Run manifest or evaluation checksum mismatch")
    if manifest["environment"]["source_sha256"] != study["source_sha256"] or result["environment"] != manifest["environment"]:
        raise ValueError("Endpoint source/environment mismatch")
    if result["initial_model_sha256"] != manifest["initial_model_sha256"]:
        raise ValueError("Endpoint initial-model identity mismatch")
    config = manifest["config"]
    if config != frozen_config(study["protocol"], condition, seed, config["data"]["data_dir"], config["device"]):
        raise ValueError("Training configuration differs from registered protocol")
    validate_endpoint(result, study["protocol"], condition, seed)
    with (directory / "evaluation.csv").open(newline="") as handle:
        endpoint_rows = [row for row in csv.DictReader(handle) if row["event"] == "endpoint"]
    expected_images = study["protocol"]["development_images"] if seed == 0 else study["protocol"]["test_images"] - study["protocol"]["development_images"]
    for role in ("fresh", "current"):
        rows = sorted((row for row in endpoint_rows if row["role"] == role), key=lambda row: int(row["draw"]))
        if [int(row["draw"]) for row in rows] != list(range(4)) or [float(row["loss"]) for row in rows] != result[f"{role}_losses"]:
            raise ValueError("Endpoint losses disagree with raw evaluation rows")
        if any(row["partition"] != result["partition"] or int(row["images"]) != expected_images for row in rows):
            raise ValueError("Endpoint evaluation population mismatch")
    return result


def pilot_records(root: Path) -> dict[str, Any]:
    return {condition: load_endpoint(root, condition, 0) for condition in CONDITIONS}


def frozen_pilot(root: Path) -> dict[str, Any]:
    manifest = study_manifest(root)
    if manifest["protocol"]["smoke"]:
        raise ValueError("Smoke studies cannot authorize confirmation")
    saved = read_json(root / "pilot_decision.json")
    computed = pilot_decision(pilot_records(root), manifest["protocol"])
    if saved != computed or not saved["passed"]:
        raise ValueError("Pilot gate is missing, changed, or failed")
    return saved


def conditions_for(root: Path, seed: int) -> list[str]:
    if seed == 0:
        return list(CONDITIONS)
    if seed not in (1, 2, 3):
        raise ValueError("Replicate outside the 18-run budget")
    pilot = frozen_pilot(root)
    interval = pilot["selected_interval"]
    return ["gaussian", "persistent", f"refresh_{interval}", f"union_{interval}"]


def run_one(root: Path, condition: str, seed: int, data_dir: str, device: str, resume: bool) -> None:
    manifest = study_manifest(root)
    if condition not in conditions_for(root, seed):
        raise ValueError("Condition not authorized by the registered stage")
    directory = run_path(root, condition, seed)
    with exclusive_lock(root / "locks" / f"{condition}_{seed}.lock"):
        if (directory / "endpoint.json").exists():
            load_endpoint(root, condition, seed)
            return
        config = frozen_config(manifest["protocol"], condition, seed, data_dir, device)
        try:
            result = train_run(config, directory, resume=resume)
            if result is None:
                raise RuntimeError("Training ended before its registered endpoint")
            immutable_json(directory / "endpoint.sha256.json", {"sha256": file_hash(directory / "endpoint.json")})
        except BaseException as exc:
            atomic_json(directory / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
            raise


def decide(root: Path) -> dict[str, Any]:
    manifest = study_manifest(root)
    spec = manifest["protocol"]
    records = pilot_records(root)
    if spec["smoke"]:
        # Synthetic data test execution and integrity, never the scientific gate.
        environments = [record["environment"] for record in records.values()]
        if any(item != environments[0] for item in environments[1:]):
            raise ValueError("Smoke environments differ")
        result = {"passed": True, "source_sha256": source_hash(), "gpu": environments[0]["gpu"] != "cpu",
                  "seconds": {key: value["seconds"] for key, value in records.items()},
                  "peak_gpu_bytes": max(value["peak_gpu_bytes"] for value in records.values()),
                  "endpoints_sha256": digest(records)}
        immutable_json(root / "smoke_receipt.json", result)
        return result
    result = pilot_decision(records, spec)
    immutable_json(root / "pilot_decision.json", result)
    if result["passed"]:
        expected = [run_path(root, condition, seed) / "endpoint.json" for seed in (1, 2, 3) for condition in conditions_for(root, seed)]
        if all(path.exists() for path in expected):
            replicates = {seed: {condition: load_endpoint(root, condition, seed) for condition in conditions_for(root, seed)} for seed in (1, 2, 3)}
            confirmation = confirmation_decision(replicates, spec, result)
            immutable_json(root / "confirmation_decision.json", confirmation)
            return confirmation
    return result


def gpu_idle(index: str) -> bool:
    output = subprocess.check_output([
        "nvidia-smi", "-i", index, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True)
    memory, utilization = [int(value.strip()) for value in output.strip().split(",")]
    return memory < 256 and utilization < 5


def execute(root: Path, stage: str, data_dir: str, gpus: list[str]) -> None:
    manifest = study_manifest(root)
    seeds = [0] if stage == "pilot" else [1, 2, 3]
    if stage == "confirmation":
        frozen_pilot(root)
    jobs = [(condition, seed) for seed in seeds for condition in conditions_for(root, seed)]
    if len(jobs) > (6 if stage == "pilot" else 12):
        raise ValueError("Run budget exceeded")
    pending = []
    for condition, seed in jobs:
        directory = run_path(root, condition, seed)
        if (directory / "failure.json").exists():
            raise RuntimeError(f"Recorded failure: {directory}")
        if (directory / "endpoint.json").exists():
            load_endpoint(root, condition, seed)
        else:
            pending.append((condition, seed))
    active: dict[str, tuple[subprocess.Popen, Any, tuple[str, int]]] = {}
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with exclusive_lock(root / "coordinator.lock"):
        try:
            while pending or active:
                for gpu, (process, log, identity) in list(active.items()):
                    code = process.poll()
                    if code is not None:
                        log.close()
                        del active[gpu]
                        if code:
                            directory = run_path(root, *identity)
                            if not (directory / "failure.json").exists():
                                atomic_json(directory / "failure.json", {"type": "WorkerExit", "exit_code": code,
                                                                         "message": "Worker exited without a complete endpoint"})
                            raise RuntimeError(f"Worker {identity} exited {code}; see {logs}")
                        load_endpoint(root, *identity)
                for gpu in gpus:
                    if not pending or gpu in active or not gpu_idle(gpu):
                        continue
                    condition, seed = pending.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUBLAS_WORKSPACE_CONFIG=":4096:8")
                    log = (logs / f"{condition}_{seed}.log").open("a", encoding="utf-8")
                    command = [sys.executable, "-m", __package__ + ".persistence_study", "run", "--root", str(root),
                               "--condition", condition, "--seed", str(seed), "--data-dir", data_dir, "--device", "cuda"]
                    if (run_path(root, condition, seed) / "checkpoints" / "recovery.pt").exists():
                        command.append("--resume")
                    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    active[gpu] = (process, log, (condition, seed))
                    print(json.dumps({"launched": condition, "seed": seed, "gpu": gpu, "pid": process.pid}), flush=True)
                atomic_json(root / "coordinator_status.json", {"stage": stage, "pending": pending,
                            "active": {gpu: {"condition": identity[0], "seed": identity[1], "pid": process.pid}
                                       for gpu, (process, _, identity) in active.items()}, "source_sha256": manifest["source_sha256"]})
                if pending or active:
                    time.sleep(10)
        except BaseException as exc:
            for process, log, _ in active.values():
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
            atomic_json(root / "coordinator_failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise
    result = decide(root)
    report(root)
    print(json.dumps(result, sort_keys=True), flush=True)


def report(root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = study_manifest(root)
    summaries, curves, times = [], [], []
    for directory in sorted((root / "runs").glob("persistence_*")):
        if not (directory / "endpoint.json").exists():
            continue
        record = read_json(directory / "endpoint.json")
        load_endpoint(root, record["condition"], record["seed"])
        summaries.append({"condition": record["condition"], "seed": record["seed"], "partition": record["partition"],
                          "fresh_mean": statistics.mean(record["fresh_losses"]), "current_mean": statistics.mean(record["current_losses"]),
                          "seconds": record["seconds"], "peak_gpu_bytes": record["peak_gpu_bytes"]})
        with (directory / "evaluation.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["seed"] == "0" and row["role"] == "fresh":
                    if row["event"] == "curve":
                        curves.append(row)
                    elif row["event"] == "timestep" and int(row["epoch"]) == manifest["protocol"]["epochs"]:
                        times.append(row)
    output = root / "report"
    output.mkdir(exist_ok=True)
    write_csv_rows(output / "endpoints.csv", summaries)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in CONDITIONS:
        selected = sorted((r for r in curves if r["condition"] == condition), key=lambda r: int(r["epoch"]))
        if selected:
            axes[0].plot([int(r["epoch"]) for r in selected], [float(r["loss"]) for r in selected], label=condition)
        selected = sorted((r for r in times if r["condition"] == condition), key=lambda r: int(r["timestep"]))
        if selected:
            axes[1].plot([int(r["timestep"]) for r in selected], [float(r["loss"]) for r in selected], label=condition)
    axes[0].set(xlabel="Epoch", ylabel="Fresh-Gaussian MSE", title="Pilot learning curves")
    axes[1].set(xlabel="Diffusion timestep", ylabel="Fresh-Gaussian MSE", title="Pilot final timestep profile")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "persistence_diagnostics.png", dpi=180)
    fig.savefig(output / "persistence_diagnostics.pdf")
    plt.close(fig)
    decisions = {name: read_json(root / f"{name}.json") for name in ("pilot_decision", "confirmation_decision") if (root / f"{name}.json").exists()}
    atomic_json(output / "decisions.json", decisions)
    text = ["# Noise Persistence Study", "", f"Completed endpoints: {len(summaries)} / 18.",
            "", "Ordering controls share exact final template exposure counts; intermediate exposure histories intentionally differ.",
            "", "A failed gate is not an equivalence result. Three confirmation replicates do not establish broad generalization.", "",
            "```json", json.dumps(decisions, indent=2, sort_keys=True), "```", ""]
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")


def quality(root: Path, device: str) -> None:
    from .evaluate_sample_quality import evaluate_run_epoch
    spec = study_manifest(root)["protocol"]
    pilot = frozen_pilot(root)
    replicates = {seed: {condition: load_endpoint(root, condition, seed) for condition in conditions_for(root, seed)} for seed in (1, 2, 3)}
    decision = confirmation_decision(replicates, spec, pilot)
    if not decision["passed"] or decision != read_json(root / "confirmation_decision.json"):
        raise ValueError("Sample-quality stage requires a verified passing confirmation")
    output = root / "sample_quality"
    output.mkdir(exist_ok=True)
    records = []
    for seed in (1, 2, 3):
        for condition in conditions_for(root, seed):
            directory = run_path(root, condition, seed)
            checkpoint = directory / "checkpoints" / f"epoch_{spec['epochs']:04d}.pt"
            if file_hash(checkpoint) != read_json(checkpoint.with_suffix(".sha256.json"))["sha256"]:
                raise ValueError("Corrupt final checkpoint")
            path = output / f"{condition}_{seed}.json"
            if path.exists():
                record = read_json(path)
            else:
                args = argparse.Namespace(device=device, seed=20261003, sample_count=50_000, real_count=50_000,
                                          sample_batch_size=128, sample_steps=50, sampler="ddim", real_split="train",
                                          grid_count=64, kid_subset_size=1000, fid_feature=2048, download_data=False)
                record = evaluate_run_epoch(directory, spec["epochs"], output, args)
                record.update(condition=condition, seed=seed, checkpoint_sha256=file_hash(checkpoint))
                validate_quality([record], spec)
                immutable_json(path, record)
            if record.get("checkpoint_sha256") != file_hash(checkpoint):
                raise ValueError("Sample-quality result belongs to another checkpoint")
            records.append(record)
    validate_quality(records, spec)
    write_csv_rows(output / "sample_quality.csv", records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "run", "execute", "decide", "report", "quality", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--stage", choices=["pilot", "confirmation"], default="pilot")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "init":
        initialize(root, smoke=args.smoke, smoke_receipt=args.smoke_receipt)
    elif args.command == "run":
        if args.condition is None:
            parser.error("run requires --condition")
        run_one(root, args.condition, args.seed, args.data_dir, args.device, args.resume)
    elif args.command == "execute":
        execute(root, args.stage, args.data_dir, args.gpus.split(","))
    elif args.command == "decide":
        print(json.dumps(decide(root), indent=2), flush=True)
    elif args.command == "report":
        report(root)
    elif args.command == "quality":
        quality(root, args.device)
    else:
        study_manifest(root)
        states = {str(path.parent.name): read_json(path) for path in (root / "runs").glob("*/status.json")}
        print(json.dumps(states, indent=2), flush=True)


if __name__ == "__main__":
    main()
