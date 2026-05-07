import json
from pathlib import Path

import pytest

from fixed_noise_diffusion.summarize_pool_dtype_control import read_eval_rows, summarize


def _write_run(root: Path, name: str, *, pool_dtype: str, gap: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    run_dir.joinpath("config.yaml").write_text(
        f"run_name: {name}\n"
        "seed: 0\n"
        "data:\n"
        "  dataset: cifar10\n"
        "noise:\n"
        "  mode: fixed_pool\n"
        "  pool_size: 1000\n"
        f"  pool_dtype: {pool_dtype}\n",
        encoding="utf-8",
    )
    run_dir.joinpath("run_metadata.json").write_text(
        json.dumps({"data": {"dataset": "cifar10"}, "noise": {"pool_size": 1000}}),
        encoding="utf-8",
    )
    run_dir.joinpath("run_summary.json").write_text("{}", encoding="utf-8")
    row = {
        "type": "eval",
        "epoch": 100,
        "step": 10,
        "train_den_loss": 0.2,
        "gaussian_den_loss": 0.2 + gap,
        "denoising_gap": gap,
        "pool_size": 1000,
    }
    run_dir.joinpath("metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_dtype_control_summary_pairs_float16_and_float32(tmp_path):
    _write_run(tmp_path, "f16_a", pool_dtype="float16", gap=0.30)
    _write_run(tmp_path, "f16_b", pool_dtype="float16", gap=0.28)
    _write_run(tmp_path, "f32_a", pool_dtype="float32", gap=0.31)
    _write_run(tmp_path, "f32_b", pool_dtype="float32", gap=0.29)

    by_dtype = {row["pool_dtype"]: row for row in summarize(read_eval_rows(tmp_path, epoch=100))}

    assert by_dtype["float16"]["n"] == 2
    assert by_dtype["float32"]["n"] == 2
    assert by_dtype["float16"]["denoising_gap_mean"] == pytest.approx(0.29)
    assert by_dtype["float32"]["denoising_gap_mean"] == pytest.approx(0.30)
    assert by_dtype["float16"]["float32_minus_float16_gap_mean"] == pytest.approx(0.01)
    assert by_dtype["float32"]["float32_minus_float16_gap_mean"] == pytest.approx(0.01)
