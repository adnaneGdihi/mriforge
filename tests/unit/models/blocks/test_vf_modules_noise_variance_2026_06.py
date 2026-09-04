"""Regression test for the 2026-06 audit fix to ``NoiseVarianceEstimator``.

The estimator ran a Python ``for b in range(B)`` loop and an
``mask_flat.sum().item()`` host sync every training step (it is a monitoring-only
scalar in vf_admm). It is now vectorized across the batch and reads the voxel
count from the boolean-index result shape (no ``.item()``).
"""
from __future__ import annotations

import torch

from spectramr.models.blocks.vf_modules import NoiseVarianceEstimator


def test_vectorized_matches_per_batch_loop() -> None:
    torch.manual_seed(0)
    est = NoiseVarianceEstimator(min_voxels=4)
    image = torch.randn(3, 2, 6, 6)
    mask = torch.zeros(1, 1, 6, 6)
    mask[..., :2, :] = 1.0  # 12 voxels

    out = est(image, mask)
    assert out.shape == (3,)

    # Reference: the previous per-batch/per-channel-var, mean-over-channels.
    mflat = mask.flatten().bool()
    ref = torch.stack(
        [image[b].reshape(2, -1)[:, mflat].var(dim=-1).mean() for b in range(3)]
    )
    assert torch.allclose(out, ref, atol=1e-5)


def test_low_voxel_guard_returns_zeros() -> None:
    est = NoiseVarianceEstimator(min_voxels=100)
    image = torch.randn(2, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0, 0] = 1.0  # 1 voxel < 100
    out = est(image, mask)
    assert torch.equal(out, torch.zeros(2))
