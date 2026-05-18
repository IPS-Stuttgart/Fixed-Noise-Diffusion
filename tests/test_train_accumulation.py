import pytest

from fixed_noise_diffusion.train import (
    _accumulation_group_size,
    _should_finish_accumulation,
)


def test_gradient_accumulation_finishes_incomplete_tail_group():
    total_batches = 5
    grad_accum_steps = 2

    assert [
        _accumulation_group_size(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [2, 2, 2, 2, 1]
    assert [
        _should_finish_accumulation(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [False, True, False, True, True]


def test_gradient_accumulation_uses_actual_tail_group_size():
    total_batches = 3
    grad_accum_steps = 8

    assert [
        _accumulation_group_size(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [3, 3, 3]
    assert [
        _should_finish_accumulation(batch_index, total_batches, grad_accum_steps)
        for batch_index in range(1, total_batches + 1)
    ] == [False, False, True]


def test_gradient_accumulation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="grad_accum_steps"):
        _accumulation_group_size(batch_index=1, total_batches=1, grad_accum_steps=0)
    with pytest.raises(ValueError, match="total_batches"):
        _accumulation_group_size(batch_index=1, total_batches=0, grad_accum_steps=1)
    with pytest.raises(ValueError, match="batch_index"):
        _accumulation_group_size(batch_index=2, total_batches=1, grad_accum_steps=1)
