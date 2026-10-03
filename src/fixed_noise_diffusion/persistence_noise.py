"""Exposure-matched noise schedules for the opt-in persistence study."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .noise import FixedPoolNoiseSampler, GaussianNoiseSampler, _positive_int


CONDITIONS = ("gaussian", "persistent", "refresh_10", "union_10", "refresh_1", "union_1")


def tensor_hash(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(memoryview(value.numpy()).cast("B")).hexdigest()


@dataclass(frozen=True)
class ExposurePlan:
    condition: str
    seed: int
    epochs: int = 100
    draws_per_epoch: int = 49_920
    pool_size: int = 1000
    image_shape: tuple[int, int, int] = (3, 32, 32)

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"Unknown persistence condition: {self.condition}")
        for name in ("epochs", "draws_per_epoch", "pool_size"):
            _positive_int(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if len(self.image_shape) != 3 or any(_positive_int("image_shape", v) != v for v in self.image_shape):
            raise ValueError("image_shape must contain three positive integers")
        if self.epochs % self.interval:
            raise ValueError("Training epochs must be divisible by the refresh interval")

    @property
    def interval(self) -> int:
        return int(self.condition.rsplit("_", 1)[1]) if "_" in self.condition else self.epochs

    @property
    def blocks(self) -> int:
        return self.epochs // self.interval

    @property
    def bank_seed(self) -> int:
        return 4242 + 1000 * self.seed

    @property
    def total_draws(self) -> int:
        return self.epochs * self.draws_per_epoch

    def sequence(self) -> torch.Tensor:
        if self.condition == "gaussian":
            return torch.empty(0, dtype=torch.long)
        local = torch.randint(self.pool_size, (self.total_draws,), generator=torch.Generator().manual_seed(200_000 + self.seed))
        if self.condition != "persistent":
            block_ids = torch.arange(self.total_draws) // (self.interval * self.draws_per_epoch)
            local.add_(block_ids * self.pool_size)
        if self.condition.startswith("union_"):
            permutation = torch.randperm(self.total_draws, generator=torch.Generator().manual_seed(300_000 + self.seed))
            local = local[permutation]
        return local


def build_bank(plan: ExposurePlan) -> torch.Tensor:
    # Blocks are generated separately, so union membership is independent of
    # construction batch sizes and exactly matches the paired refresh run.
    bank = torch.empty((plan.epochs * plan.pool_size, *plan.image_shape), dtype=torch.float32)
    for block in range(plan.epochs):
        generator = torch.Generator().manual_seed(plan.bank_seed + block)
        bank[block * plan.pool_size : (block + 1) * plan.pool_size] = torch.randn(
            (plan.pool_size, *plan.image_shape), generator=generator, dtype=torch.float32
        )
    return bank


class ScheduledNoiseSampler:
    def __init__(self, plan: ExposurePlan, device: torch.device, bank: torch.Tensor | None = None) -> None:
        self.plan = plan
        self.device = device
        self.sequence = plan.sequence()
        self.sequence_sha256 = tensor_hash(self.sequence)
        self.bank = None if plan.condition == "gaussian" else (build_bank(plan) if bank is None else bank)
        if self.bank is not None:
            expected = (plan.epochs * plan.pool_size, *plan.image_shape)
            if tuple(self.bank.shape) != expected or self.bank.dtype != torch.float32 or self.bank.device.type != "cpu":
                raise ValueError("Unexpected bank shape, dtype, or device")
        self.block_hashes = [] if self.bank is None else [
            tensor_hash(self.bank[i * plan.pool_size : (i + 1) * plan.pool_size]) for i in range(plan.epochs)
        ]
        self.cursor = 0
        self.epoch = 0
        self.counts = torch.zeros(plan.epochs * plan.pool_size, dtype=torch.long)
        self.gaussian = GaussianNoiseSampler(plan.image_shape, device, 500_000 + plan.seed)

    def begin_epoch(self, epoch: int) -> None:
        if not 1 <= epoch <= self.plan.epochs or self.cursor != (epoch - 1) * self.plan.draws_per_epoch:
            raise ValueError("Epoch boundary does not match the replay cursor")
        self.epoch = epoch

    @property
    def active_block(self) -> int:
        return max(0, self.epoch - 1) // self.plan.interval if self.plan.condition.startswith("refresh_") else 0

    def sample(self, batch_size: int) -> torch.Tensor:
        batch_size = _positive_int("batch_size", batch_size)
        end = self.cursor + batch_size
        if self.epoch == 0 or end > self.epoch * self.plan.draws_per_epoch or end > self.plan.total_draws:
            raise ValueError("Noise draw crosses an epoch or study boundary")
        if self.bank is None:
            result = self.gaussian.sample(batch_size)
        else:
            indices = self.sequence[self.cursor:end]
            self.counts.scatter_add_(0, indices, torch.ones_like(indices))
            result = self.bank.index_select(0, indices).to(self.device)
        self.cursor = end
        return result

    def evaluation_sampler(self, role: str, seed: int, block: int | None = None):
        if role == "fresh" or self.bank is None:
            return GaussianNoiseSampler(self.plan.image_shape, self.device, seed)
        active = self.active_block if block is None else block
        if not 0 <= active < self.plan.blocks:
            raise ValueError("Invalid active pool block")
        size = self.plan.pool_size * (self.plan.blocks if self.plan.condition.startswith("union_") else 1)
        pool = None
        pool_seed = self.plan.bank_seed + 1_000_003
        if role == "current":
            start = active * self.plan.pool_size
            pool = self.bank[:size] if self.plan.condition.startswith("union_") else self.bank[start:start + size]
            pool_seed = self.plan.bank_seed + active
        elif role == "retired":
            if not self.plan.condition.startswith("refresh_") or active < 1:
                raise ValueError("This condition has no retired pool")
            pool = self.bank[(active - 1) * size : active * size]
            pool_seed = self.plan.bank_seed + active - 1
        elif role != "unseen":
            raise ValueError(f"Unknown evaluation role: {role}")
        return FixedPoolNoiseSampler(
            self.plan.image_shape, self.device, size, pool_seed, seed,
            dtype="float32", existing_pool=pool, track_exposure=False,
        )

    def fingerprint(self) -> dict[str, Any]:
        blocks = list(range(self.plan.blocks)) if self.plan.condition.startswith("union_") else [self.active_block]
        return {
            "fingerprint_version": 2,
            "plan": asdict(self.plan),
            "sequence_sha256": self.sequence_sha256,
            "block_sha256": self.block_hashes,
            "active_blocks": [] if self.bank is None else blocks,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "exposure_sha256": tensor_hash(self.counts),
            "unique_templates_seen": int((self.counts > 0).sum()),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint(), "counts": self.counts.clone(), "gaussian_rng": self.gaussian.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        identity = state["fingerprint"]
        own = self.fingerprint()
        for key in ("fingerprint_version", "plan", "sequence_sha256", "block_sha256"):
            if identity[key] != own[key]:
                raise ValueError(f"Noise recovery identity mismatch: {key}")
        cursor, epoch = identity["cursor"], identity["epoch"]
        if not 0 <= epoch <= self.plan.epochs or cursor != epoch * self.plan.draws_per_epoch:
            raise ValueError("Recovery requires a completed epoch")
        expected = torch.bincount(self.sequence[:cursor], minlength=len(self.counts))
        if not torch.equal(state["counts"], expected):
            raise ValueError("Corrupt exposure counters")
        self.counts = state["counts"].clone()
        self.cursor, self.epoch = cursor, epoch
        self.gaussian.generator.set_state(state["gaussian_rng"])
        if self.fingerprint() != identity:
            raise ValueError("Corrupt active-pool fingerprint")
