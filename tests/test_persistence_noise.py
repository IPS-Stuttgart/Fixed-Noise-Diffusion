from dataclasses import replace

import pytest
import torch

from fixed_noise_diffusion.persistence_noise import ExposurePlan, ScheduledNoiseSampler, build_bank, tensor_hash


def small_plan(condition="refresh_1"):
    return ExposurePlan(condition, 0, epochs=10, draws_per_epoch=12, pool_size=3, image_shape=(1, 2, 2))


@pytest.mark.parametrize("interval", [1, 10])
def test_union_is_exact_permutation_with_identical_bank(interval):
    refresh = small_plan(f"refresh_{interval}")
    union = replace(refresh, condition=f"union_{interval}")
    assert torch.equal(refresh.sequence().sort().values, union.sequence().sort().values)
    assert not torch.equal(refresh.sequence(), union.sequence())
    assert torch.equal(build_bank(refresh), build_bank(union))
    assert torch.equal(build_bank(refresh)[:3], build_bank(replace(refresh, condition="persistent"))[:3])


def test_refresh_boundaries_and_end_budget():
    sampler = ScheduledNoiseSampler(small_plan(), torch.device("cpu"))
    with pytest.raises(ValueError, match="boundary"):
        sampler.sample(1)
    for epoch in range(1, 11):
        sampler.begin_epoch(epoch)
        assert sampler.active_block == epoch - 1
        sampler.sample(12)
        with pytest.raises(ValueError, match="boundary"):
            sampler.sample(1)
    with pytest.raises(ValueError, match="boundary"):
        sampler.begin_epoch(11)
    assert sampler.active_block == 9
    assert int(sampler.counts.sum()) == 120


def test_reconstruction_counts_and_current_retired_unseen():
    original = ScheduledNoiseSampler(small_plan(), torch.device("cpu"))
    original.begin_epoch(1)
    original.sample(12)
    state = original.state_dict()
    restored = ScheduledNoiseSampler(small_plan(), torch.device("cpu"))
    restored.load_state_dict(state)
    for sampler in (original, restored):
        sampler.begin_epoch(2)
    assert torch.equal(original.sample(12), restored.sample(12))
    pools = [original.evaluation_sampler(role, 11).pool for role in ("current", "retired", "unseen")]
    assert len({tensor_hash(pool) for pool in pools}) == 3
    assert torch.equal(pools[0], original.bank[3:6])
    assert torch.equal(pools[1], original.bank[:3])
    assert original.fingerprint() == restored.fingerprint()
    before = original.fingerprint()
    original.evaluation_sampler("current", 33).sample(50)
    assert original.fingerprint() == before


def test_corrupt_recovery_and_foreign_pool_rejected():
    sampler = ScheduledNoiseSampler(small_plan(), torch.device("cpu"))
    sampler.begin_epoch(1)
    sampler.sample(12)
    state = sampler.state_dict()
    state["counts"][0] += 1
    with pytest.raises(ValueError, match="exposure"):
        sampler.load_state_dict(state)
    foreign = ScheduledNoiseSampler(replace(small_plan(), seed=1), torch.device("cpu"))
    with pytest.raises(ValueError, match="identity"):
        foreign.load_state_dict(sampler.state_dict())


def test_gaussian_recovery_reproduces_private_stream():
    first = ScheduledNoiseSampler(small_plan("gaussian"), torch.device("cpu"))
    first.begin_epoch(1)
    first.sample(12)
    second = ScheduledNoiseSampler(small_plan("gaussian"), torch.device("cpu"))
    second.load_state_dict(first.state_dict())
    first.begin_epoch(2)
    second.begin_epoch(2)
    assert torch.equal(first.sample(12), second.sample(12))


def test_plan_rejects_unknown_or_incompatible_schedule():
    with pytest.raises(ValueError):
        small_plan("other")
    with pytest.raises(ValueError, match="divisible"):
        replace(small_plan("refresh_10"), epochs=11)
