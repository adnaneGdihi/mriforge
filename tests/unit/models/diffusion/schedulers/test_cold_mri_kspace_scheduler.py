"""Tests for :mod:`spectramr.models.diffusion.schedulers.cold_mri_kspace_scheduler`.

``degrade`` builds ``k_space`` as ``(B, H, W)`` and asks the accelerator for a mask
with ``k_space.shape[1:]`` — a 2-tuple, so the contractual return is ``(1, H, W)``,
already broadcastable over the batch. The scheduler nonetheless unsqueezed it to
``(1, 1, H, W)``, which broadcasts ``(B, H, W)`` up to ``(1, B, H, W)`` and promotes
the batch axis into a channel axis. It escaped notice only because the trajectory
families used to return a rank-2 mask here and skipped the branch entirely.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.sampling import create_kspace_accelerator
from spectramr.models.diffusion.schedulers.cold_mri_kspace_scheduler import (
    ColdMRIKSpaceScheduler,
)

MATRIX = 32
BATCH = 3
_CPU = torch.device("cpu")


def _scheduler(family: str) -> ColdMRIKSpaceScheduler:
    return ColdMRIKSpaceScheduler(
        num_train_timesteps=8,
        accelerator=create_kspace_accelerator(
            acceleration_type=family,
            num_timesteps=8,
            base_acceleration=2.0,
            max_acceleration=8.0,
            center_fraction=0.08,
            min_center_fraction=0.02,
            seed=42,
            acceleration_schedule="linear",
        ),
    )


@pytest.mark.parametrize("family", ["random_cartesian", "radial", "golden_angle"])
def test_degrade_preserves_the_input_shape(family: str) -> None:
    """The batch axis must survive degradation, for every family."""
    image = torch.randn(BATCH, 2, MATRIX, MATRIX)
    degraded = _scheduler(family).degrade(image, 4)
    assert degraded.shape == image.shape


@pytest.mark.parametrize("family", ["random_cartesian", "radial", "golden_angle"])
def test_degrade_is_independent_per_batch_item(family: str) -> None:
    """Item ``i`` of the output depends only on item ``i`` of the input.

    A spurious leading axis broadcasts every item against every other, which shows
    up here as a batch whose entries no longer track their own inputs.
    """
    scheduler = _scheduler(family)
    image = torch.randn(BATCH, 2, MATRIX, MATRIX)
    batched = scheduler.degrade(image, 4)
    for i in range(BATCH):
        alone = scheduler.degrade(image[i : i + 1], 4)
        assert torch.allclose(batched[i], alone[0], atol=1e-5)


def test_rank_violating_accelerator_is_rejected() -> None:
    """A mask that breaks the rank contract raises here rather than broadcasting."""
    scheduler = _scheduler("random_cartesian")
    inner = scheduler.accelerator
    original = inner.get_acceleration_mask

    def _rank_two(shape, t, device=_CPU):
        return original(shape, t, device=device)[0]

    inner.get_acceleration_mask = _rank_two
    with pytest.raises(ValueError, match=r"expected \(channels, height, width\)"):
        scheduler.degrade(torch.randn(BATCH, 2, MATRIX, MATRIX), 4)
