r"""Physics tests for the DL-BAE per-voxel dispersion decoder.

Targets ``spectramr.infrastructure.physics.dispersion.dispersion_rates_voxelwise``.

The voxelwise evaluator is the decoder half of DL-BAE: it takes a *latent map*
and evaluates the BPP dispersion law at every field, producing ``[B, M, H, W]``.
It must agree exactly with the established scalar path
(:func:`dispersion_r1` / :func:`dispersion_r2`) voxel-by-voxel -- the scalar path
is the physics SSOT, so any divergence means the decoder is silently modelling
something else.

Also asserted: the monotonicity property :math:`\partial T_1/\partial B_0\ge0`
that the ``dispersion_monotonicity`` loss exists to protect.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.dispersion import (
    dispersion_r1,
    dispersion_r2,
    dispersion_rates_voxelwise,
    dispersion_t1,
)

pytestmark = pytest.mark.physics


def _latent(batch: int, pools: int, h: int, w: int, seed: int = 0):
    torch.manual_seed(seed)
    return {
        "a0": torch.rand(batch, 1, h, w, dtype=torch.float64) + 0.1,
        "c0": torch.rand(batch, 1, h, w, dtype=torch.float64) + 0.1,
        "b": torch.rand(batch, pools, h, w, dtype=torch.float64) + 0.1,
        "tau_c": torch.rand(batch, pools, h, w, dtype=torch.float64) * 1e-8 + 1e-9,
    }


@pytest.mark.parametrize("pools", [1, 2])
def test_voxelwise_matches_scalar_path_exactly(pools: int) -> None:
    """Every voxel must reproduce the scalar SSOT evaluation bit-for-bit."""
    b0 = torch.tensor([0.055, 0.3, 1.5, 3.0, 7.0], dtype=torch.float64)
    lat = _latent(2, pools, 4, 5)
    r1, r2 = dispersion_rates_voxelwise(b0, **lat)
    assert r1.shape == (2, b0.numel(), 4, 5)

    for bi, hi, wi in ((0, 0, 0), (1, 2, 3), (1, 3, 4)):
        ref1 = dispersion_r1(
            b0,
            a0=float(lat["a0"][bi, 0, hi, wi]),
            b=lat["b"][bi, :, hi, wi],
            tau_c=lat["tau_c"][bi, :, hi, wi],
        )
        ref2 = dispersion_r2(
            b0,
            c0=float(lat["c0"][bi, 0, hi, wi]),
            b=lat["b"][bi, :, hi, wi],
            tau_c=lat["tau_c"][bi, :, hi, wi],
        )
        assert torch.allclose(r1[bi, :, hi, wi], ref1, atol=0.0, rtol=0.0)
        assert torch.allclose(r2[bi, :, hi, wi], ref2, atol=0.0, rtol=0.0)


def test_pool_count_mismatch_raises() -> None:
    """A b/tau_c pool-count disagreement is a wiring bug, not something to broadcast."""
    b0 = torch.tensor([0.3, 1.5, 3.0], dtype=torch.float64)
    lat = _latent(1, 2, 3, 3)
    lat["tau_c"] = lat["tau_c"][:, :1]
    with pytest.raises(ValueError, match="share shape"):
        dispersion_rates_voxelwise(b0, **lat)


def test_t1_is_monotone_in_field_for_physiological_tau() -> None:
    """BPP predicts T1 non-decreasing in B0 over the physiological tau_c range."""
    b0 = torch.tensor([0.055, 0.3, 1.5, 3.0, 7.0], dtype=torch.float64)
    t1 = dispersion_t1(
        b0,
        a0=0.2,
        b=torch.tensor([1e8], dtype=torch.float64),
        tau_c=torch.tensor([5e-9], dtype=torch.float64),
    )
    assert torch.all(t1.diff() >= -1e-12), t1


def test_voxelwise_gradient_flows_to_latent() -> None:
    """The decoder must be differentiable end-to-end or DL-BAE cannot train."""
    b0 = torch.tensor([0.3, 1.5, 3.0], dtype=torch.float64)
    lat = _latent(1, 1, 2, 2)
    for v in lat.values():
        v.requires_grad_(True)
    # Both rates: R1 carries a0, R2 carries c0, so a single-rate backward would
    # legitimately leave the other baseline without a gradient.
    r1, r2 = dispersion_rates_voxelwise(b0, **lat)
    (r1.sum() + r2.sum()).backward()
    for name, v in lat.items():
        assert v.grad is not None and torch.isfinite(v.grad).all(), name
