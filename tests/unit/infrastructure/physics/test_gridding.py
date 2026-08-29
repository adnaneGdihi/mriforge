"""Tests for density-compensation and gridding helpers.

Targets ``mriforge.infrastructure.physics.gridding``. ``RadialDCF`` and
``IterativeDCF`` produce per-sample density-compensation weights for
non-Cartesian k-space; ``KaiserBesselKernel`` is the apodization helper
used in analytic gridding inside iterative solvers.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.gridding import (
    IterativeDCF,
    KaiserBesselKernel,
    RadialDCF,
)


# ---------------------------------------------------------------------------
# RadialDCF
# ---------------------------------------------------------------------------


def test_radial_dcf_invalid_num_spokes_raises() -> None:
    """num_spokes < 1 fails loud."""
    with pytest.raises(ValueError, match="num_spokes"):
        RadialDCF(num_spokes=0, samples_per_spoke=64)


def test_radial_dcf_invalid_samples_per_spoke_raises() -> None:
    """samples_per_spoke < 2 fails loud."""
    with pytest.raises(ValueError, match="samples_per_spoke"):
        RadialDCF(num_spokes=64, samples_per_spoke=1)


def test_radial_dcf_traj_wrong_last_dim_raises() -> None:
    """Trajectory must end in 2 (kx, ky)."""
    dcf = RadialDCF(num_spokes=8, samples_per_spoke=16)
    bad_traj = torch.zeros(8, 16, 3)  # last dim 3, not 2
    with pytest.raises(ValueError, match="traj must end in 2"):
        dcf(bad_traj)


def test_radial_dcf_weights_increase_with_radius() -> None:
    """Radial DCF weight ∝ |k_r|; weights at large radius exceed central."""
    dcf = RadialDCF(num_spokes=4, samples_per_spoke=8)
    # Build a radial trajectory: 4 spokes × 8 samples each.
    spokes = 4
    samples = 8
    angles = torch.linspace(0, math.pi, spokes + 1)[:-1]
    radii = torch.linspace(0, math.pi, samples)
    kx = torch.outer(torch.cos(angles), radii)  # [4, 8]
    ky = torch.outer(torch.sin(angles), radii)
    traj = torch.stack([kx, ky], dim=-1)  # [4, 8, 2]
    weights = dcf(traj)
    # Per spoke, weights should be monotonically non-decreasing with sample index
    # (sample 0 is at origin → uses central_w; later samples have |k_r| > 0).
    for s in range(spokes):
        # Outer samples (idx=samples-1) > inner samples (idx=1).
        assert weights[s, samples - 1] > weights[s, 1]


def test_radial_dcf_normalised_to_unit_mean() -> None:
    """The trailing normalisation gives weights with mean ≈ 1."""
    dcf = RadialDCF(num_spokes=8, samples_per_spoke=16)
    traj = torch.randn(8, 16, 2) * 0.5
    weights = dcf(traj)
    assert abs(weights.mean().item() - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# IterativeDCF
# ---------------------------------------------------------------------------


def test_iterative_dcf_invalid_kernel_width_raises() -> None:
    """kernel_width ≤ 0 fails loud."""
    with pytest.raises(ValueError, match="kernel_width and beta"):
        IterativeDCF(kernel_width=0.0)


def test_iterative_dcf_invalid_beta_raises() -> None:
    """beta ≤ 0 fails loud."""
    with pytest.raises(ValueError, match="kernel_width and beta"):
        IterativeDCF(kernel_width=4.0, beta=-1.0)


def test_iterative_dcf_traj_wrong_dim_raises() -> None:
    """Trajectory must be exactly [N, 2]."""
    dcf = IterativeDCF(num_iters=2)
    with pytest.raises(ValueError, match=r"traj must be \[N, 2\]"):
        dcf(torch.zeros(8, 16, 2))  # 3D, not 2D


def test_iterative_dcf_returns_finite_weights() -> None:
    """Pipe-Menon iteration produces finite, positive weights."""
    dcf = IterativeDCF(num_iters=3)
    traj = torch.randn(64, 2) * 0.1
    weights = dcf(traj)
    assert weights.shape == (64,)
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()


def test_iterative_dcf_normalised_to_unit_mean() -> None:
    """Trailing normalisation gives mean ≈ 1."""
    dcf = IterativeDCF(num_iters=3)
    traj = torch.randn(32, 2) * 0.1
    weights = dcf(traj)
    assert abs(weights.mean().item() - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# KaiserBesselKernel
# ---------------------------------------------------------------------------


def test_kb_kernel_invalid_args_raise() -> None:
    """Invalid kernel_width / beta fail loud."""
    with pytest.raises(ValueError):
        KaiserBesselKernel(kernel_width=-1.0)
    with pytest.raises(ValueError):
        KaiserBesselKernel(beta=0.0)


def test_kb_kernel_zero_outside_support() -> None:
    """Kernel is zero for distance > kernel_width / 2."""
    kw = 4.0
    kb = KaiserBesselKernel(kernel_width=kw)
    far = torch.full((10,), kw)  # 2 * far / kw == 2 → outside support
    out = kb(far)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-7)


def test_kb_kernel_peaks_at_zero_distance() -> None:
    """Maximum kernel value occurs at distance = 0."""
    kb = KaiserBesselKernel(kernel_width=4.0)
    distances = torch.linspace(0, 2.0, 21)  # within support
    out = kb(distances)
    assert torch.argmax(out).item() == 0


def test_kb_kernel_normalization_property_is_positive() -> None:
    """The closed-form L1 normaliser is positive."""
    kb = KaiserBesselKernel(kernel_width=4.0, beta=13.9)
    assert kb.normalization > 0


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leading_shape", [(64,), (8, 16), (4, 8, 16)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_radial_dcf_preserves_leading_shape(leading_shape: tuple[int, ...]) -> None:
    """RadialDCF outputs match the leading shape of the trajectory."""
    dcf = RadialDCF(num_spokes=8, samples_per_spoke=16)
    traj = torch.randn(*leading_shape, 2) * 0.1
    weights = dcf(traj)
    assert weights.shape == leading_shape
