import csv
import json
from pathlib import Path

from fixed_noise_diffusion.summarize_pool_dtype_control import read_eval_rows, summarize, write_csv


def _write_run(root: Path, name: str, *, seed: int, pool_size: int, pool_dtype: str, gap: float) -> None:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                f"run_name: {name}",
                f"seed: {seed}",
                "data:",
                "  dataset: cifar10",
                "noise:",
                "  mode: fixed_pool",
                f"  pool_size: {pool_size}",
                f"  pool_dtype: {pool_dtype}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_name": name,
                "seed": seed,
                "data": {"dataset": "cifar10"},
                "noise": {"pool_size": pool_size},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_summary.json").write_text(
        json.dumps({"run_name": name, "seed": seed}), encoding="utf-8"
    )
    eval_row = {
        "type": "eval",
        "epoch": 100,
        "step": 39000,
        "train_den_loss": 0.2,
        "gaussian_den_loss": 0.2 + gap,
        "denoising_gap": gap,
        "pool_size": pool_size,
    }
    (run_dir / "metrics.jsonl").write_text(json.dumps(eval_row) + "\n", encoding="utf-8")


def test_dtype_control_summary_pairs_float16_and_float32(tmp_path):
    _write_run(tmp_path, "pool1k-f16-seed0", seed=0, pool_size=1000, pool_dtype="float16", gap=0.30)
    _write_run(tmp_path, "pool1k-f16-seed1", seed=1, pool_size=1000, pool_dtype="float16", gap=0.28)
    _write_run(tmp_path, "pool1k-f32-seed0", seed=0, pool_size=1000, pool_dtype="float32", gap=0.31)
    _write_run(tmp_path, "pool1k-f32-seed1", seed=1, pool_size=1000, pool_dtype="float32", gap=0.29)

    eval_rows = read_eval_rows(tmp_path, epoch=100)
    summary_rows = summarize(eval_rows)

    assert len(eval_rows) == 4
    assert len(summary_rows) == 2
    by_dtype = {row["pool_dtype"]: row for row in summary_rows}
    assert by_dtype["float16"]["n"] == 2
    assert by_dtype["float32"]["n"] == 2
    assert by_dtype["float16"]["denoising_gap_mean"] == 0.29
    assert by_dtype["float32"]["denoising_gap_mean"] == 0.30
    assert by_dtype["float16"]["float32_minus_float16_gap_mean"] == by_dtype["float32"]["float32_minus_float16_gap_mean"]
    assert abs(by_dtype["float32"]["float32_minus_float16_gap_mean"] - 0.01) < 1e-12


def test_dtype_control_summary_writes_expected_columns(tmp_path):
    rows = [
        {
            "dataset": "cifar10",
            "pool_size": 1000,
            "pool_dtype": "float16",
            "epoch": 100,
            "n": 1,
            "denoising_gap_mean": 0.3,
            "denoising_gap_std": 0.0,
            "denoising_gap_sem": 0.0,
            "float32_minus_float16_gap_mean": "",
        }
    ]
    output = tmp_path / "summary.csv"
    columns = [
        "dataset",
        "pool_size",
        "pool_dtype",
        "epoch",
        "n",
        "denoising_gap_mean",
        "denoising_gap_std",
        "denoising_gap_sem",
        "float32_minus_float16_gap_mean",
    ]
    write_csv(output, rows, columns)

    with output.open(newline="", encoding="utf-8") as handle:
        loaded = list(csv.DictReader(handle))
    assert loaded == [
        {
            "dataset": "cifar10",
            "pool_size": "1000",
            "pool_dtype": "float16",
            "epoch": "100",
            "n": "1",
            "denoising_gap_mean": "0.3",
            "denoising_gap_std": "0",
            "denoising_gap_sem": "0",
            "float32_minus_float16_gap_mean": "",
        }
    ]
