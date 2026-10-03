from copy import deepcopy

import pytest

from fixed_noise_diffusion.persistence_noise import CONDITIONS
from fixed_noise_diffusion.persistence_protocol import (
    confirmation_decision, digest, frozen_config, immutable_json, partitions, pilot_decision, protocol, validate_quality,
)


def records(seed=0, interval=None, effect=0.01):
    spec = protocol()
    conditions = CONDITIONS if interval is None else ["gaussian", "persistent", f"refresh_{interval}", f"union_{interval}"]
    result = {}
    for condition in conditions:
        fresh = 0.3 if condition == "persistent" else 0.06
        if condition.startswith("refresh_"):
            fresh += effect
        result[condition] = {
            "condition": condition, "seed": seed, "epoch": 100, "step": 39000,
            "partition": "development" if seed == 0 else "confirmation", "protocol_sha256": digest(spec),
            "fresh_losses": [fresh] * 4, "current_losses": [0.01 if condition == "persistent" else 0.05] * 4,
            "environment": {"host": "same", "gpu": "same"},
            "initial_model_sha256": "same",
            "noise": {"block_sha256": ["matched"], "exposure_sha256": "matched"},
        }
    return result


def test_partition_is_fixed_disjoint_and_exhaustive():
    dev, final = partitions(protocol())
    assert (len(dev), len(final)) == (2048, 7952)
    assert not set(dev).intersection(final)
    assert sorted(dev + final) == list(range(10000))
    assert (dev, final) == partitions(protocol())


def test_frozen_configuration_overrides_legacy_defaults():
    config = frozen_config(protocol(), "refresh_1", 0, "/data", "cuda")
    assert config["diffusion"]["beta_schedule"] == "cosine"
    assert config["noise"]["pool_dtype"] == "float32"
    assert config["training"]["amp"] is False
    assert config["evaluation"]["fid_feature"] == 2048


def test_pilot_tie_prefers_ten_epochs_and_freezes_direction():
    decision = pilot_decision(records(), protocol())
    assert decision["passed"]
    assert decision["selected_interval"] == 10
    assert decision["selected_direction"] == 1
    assert pilot_decision(records(effect=-0.01), protocol())["selected_direction"] == -1


def test_gap_closure_without_ordering_effect_does_not_pass():
    assert not pilot_decision(records(effect=0), protocol())["passed"]
    weak = records()
    weak["persistent"]["fresh_losses"] = [0.061] * 4
    assert not pilot_decision(weak, protocol())["passed"]


def test_mc_error_and_protocol_integrity_are_enforced():
    noisy = records()
    for condition in ("refresh_1", "refresh_10"):
        noisy[condition]["fresh_losses"] = [0.02, 0.02, 0.2, 0.2]
    assert not pilot_decision(noisy, protocol())["passed"]
    invalid = records()
    invalid["union_1"]["noise"]["exposure_sha256"] = "different"
    with pytest.raises(ValueError, match="exposure"):
        pilot_decision(invalid, protocol())
    invalid = records()
    invalid["persistent"]["environment"]["gpu"] = "different"
    with pytest.raises(ValueError, match="hardware"):
        pilot_decision(invalid, protocol())
    invalid = records()
    invalid["persistent"]["fresh_losses"][0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        pilot_decision(invalid, protocol())


def test_confirmation_requires_fresh_seeds_direction_margin_and_interval():
    spec = protocol()
    pilot = pilot_decision(records(), spec)
    replicates = {seed: records(seed, interval=10, effect=0.01 + seed * 0.0001) for seed in (1, 2, 3)}
    assert confirmation_decision(replicates, spec, pilot)["passed"]
    replicates[3] = records(3, interval=10, effect=-0.01)
    assert not confirmation_decision(replicates, spec, pilot)["passed"]
    with pytest.raises(ValueError, match="three fresh"):
        confirmation_decision({1: replicates[1]}, spec, pilot)


def test_fid_protocol_mismatch_is_rejected():
    record = dict(protocol()["fid"], fid=42.0)
    validate_quality([record], protocol())
    for key, value in (("fid_feature", 64), ("fake_count", 2048), ("real_split", "val")):
        invalid = deepcopy(record)
        invalid[key] = value
        with pytest.raises(ValueError, match="FID protocol"):
            validate_quality([record, invalid], protocol())


def test_immutable_json_normalizes_tuples_and_rejects_changes(tmp_path):
    path = tmp_path / "record.json"
    immutable_json(path, {"shape": (1, 2, 3)})
    immutable_json(path, {"shape": (1, 2, 3)})
    with pytest.raises(ValueError, match="Immutable"):
        immutable_json(path, {"shape": (1, 2, 4)})
