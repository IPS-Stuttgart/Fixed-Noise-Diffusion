from copy import deepcopy

import pytest
import torch

from fixed_noise_diffusion.noise import make_noise_sampler
from fixed_noise_diffusion.persistence_noise import ExposurePlan, ScheduledNoiseSampler, tensor_hash
from fixed_noise_diffusion.persistence_protocol import frozen_config, protocol
from fixed_noise_diffusion.persistence_train import capture_rng, evaluate_snapshot, train_run
from fixed_noise_diffusion.data import make_dataloaders
from fixed_noise_diffusion.model import build_model
from fixed_noise_diffusion.diffusion import GaussianDiffusion


def tiny_config():
    spec = protocol(smoke=True)
    spec.update(epochs=2, train_images=4, test_images=4, development_images=2,
                batch_size=2, pool_size=2, image_shape=[1, 4, 4], curve_epochs=[1, 2],
                timestep_epochs=[], boundary_epochs=[1], timesteps=[0, 3])
    config = frozen_config(spec, "refresh_1", 0, "unused", "cpu")
    config["model"].update(base_channels=8, channel_mults=[1], time_emb_dim=16)
    config["diffusion"].update(num_timesteps=4)
    config["data"]["eval_batch_size"] = 2
    return config


def test_evaluation_preserves_rng_model_mode_and_exposures():
    config = tiny_config()
    model = build_model(config).train()
    diffusion = GaussianDiffusion.from_config(config, torch.device("cpu"))
    dataset = make_dataloaders(config).val.dataset
    sampler = ScheduledNoiseSampler(ExposurePlan("refresh_1", 0, 2, 4, 2, (1, 4, 4)), torch.device("cpu"))
    sampler.begin_epoch(1)
    sampler.sample(4)
    before = capture_rng()
    fingerprint = sampler.fingerprint()
    rows = evaluate_snapshot(model, diffusion, dataset, [0, 1], sampler, config,
                             epoch=1, event="test", partition="development", draws=2)
    after = capture_rng()
    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"]
    assert before["numpy"] == after["numpy"]
    assert model.training
    assert sampler.fingerprint() == fingerprint
    assert len(rows) == 6


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_epoch_recovery_matches_uninterrupted_training(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    config = tiny_config()
    config["device"] = device
    full = train_run(config, tmp_path / "full")
    assert train_run(config, tmp_path / "resumed", stop_after_epoch=1) is None
    resumed = train_run(config, tmp_path / "resumed", resume=True)
    assert full["fresh_losses"] == resumed["fresh_losses"]
    assert full["current_losses"] == resumed["current_losses"]
    assert full["noise"] == resumed["noise"]
    first = torch.load(tmp_path / "full/checkpoints/recovery.pt", weights_only=True)
    second = torch.load(tmp_path / "resumed/checkpoints/recovery.pt", weights_only=True)
    assert {key: tensor_hash(value) for key, value in first["model"].items()} == {key: tensor_hash(value) for key, value in second["model"].items()}
    with pytest.raises(FileExistsError):
        train_run(config, tmp_path / "full")
    (tmp_path / "full/evaluation.csv").unlink()
    (tmp_path / "full/endpoint.json").unlink()
    reconstructed = train_run(config, tmp_path / "full", resume=True)
    assert reconstructed["fresh_losses"] == full["fresh_losses"]


def test_legacy_factory_rejects_study_config_but_keeps_legacy_behavior():
    config = tiny_config()
    with pytest.raises(ValueError, match="reconstructed"):
        make_noise_sampler(config, torch.device("cpu"))
    legacy = deepcopy(config)
    legacy.pop("persistence")
    with pytest.raises(ValueError, match="reconstructed"):
        make_noise_sampler(legacy, torch.device("cpu"))
    legacy["noise"].pop("schedule")
    assert make_noise_sampler(legacy, torch.device("cpu")).sample(2).shape == (2, 1, 4, 4)
