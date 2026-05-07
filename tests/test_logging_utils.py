import csv

from fixed_noise_diffusion.logging_utils import MetricLogger


def test_metric_logger_writes_denoising_eval_timestep_seed_to_csv(tmp_path):
    logger = MetricLogger(tmp_path)
    logger.log(
        {
            "type": "eval",
            "epoch": 7,
            "step": 123,
            "denoising_eval_timestep_seed": 30010,
        }
    )

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["denoising_eval_timestep_seed"] == "30010"
