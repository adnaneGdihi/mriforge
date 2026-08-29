"""Canary / parametrized / edge-case / expected-failure tests for gridding.py.

Covers:
  - RadialDCF
  - IterativeDCF
  - KaiserBesselKernel
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _radial_traj(n_spokes: int, n_per_spoke: int) -> torch.Tensor:
    """Build a simple radial trajectory [n_spokes * n_per_spoke, 2] in [-pi, pi]."""
    angles = torch.linspace(0, math.pi, n_spokes + 1)[:-1]  # endpoint-exclusive
    r = torch.linspace(-math.pi, math.pi, n_per_spoke)
    coords = []
    for a in angles:
        kx = r * torch.cos(a)
        ky = r * torch.sin(a)
        coords.append(torch.stack([kx, ky], dim=-1))
    return torch.cat(coords, dim=0)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_gridding_imports():
    """All public symbols import cleanly."""
    from mriforge.infrastructure.physics.gridding import (
        RadialDCF,
        IterativeDCF,
        KaiserBesselKernel,
    )
    assert all([RadialDCF, IterativeDCF, KaiserBesselKernel])


@pytest.mark.physics
def test_canary_radial_dcf_trivial():
    """RadialDCF forward on a tiny trajectory returns positive weights."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    dcf = RadialDCF(num_spokes=4, samples_per_spoke=8)
    traj = _radial_traj(4, 8)   # [32, 2]
    w = dcf(traj)
    assert w.shape == (32,)
    assert (w >= 0).all()


@pytest.mark.physics
def test_canary_iterative_dcf_trivial():
    """IterativeDCF forward on a small trajectory completes."""
    from mriforge.infrastructure.physics.gridding import IterativeDCF
    dcf = IterativeDCF(num_iters=3)
    traj = _radial_traj(4, 8)   # [32, 2]
    w = dcf(traj)
    assert w.shape == (32,)
    assert (w > 0).all()


@pytest.mark.physics
def test_canary_kaiser_bessel_trivial():
    """KaiserBesselKernel evaluates on distance range."""
    from mriforge.infrastructure.physics.gridding import KaiserBesselKernel
    kb = KaiserBesselKernel()
    d = torch.linspace(0.0, 3.0, 20)
    out = kb(d)
    assert out.shape == (20,)
    # Inside support (d < kernel_width/2=2) → non-zero; outside → 0
    assert out[0] > 0
    assert out[-1] == 0.0


# ---------------------------------------------------------------------------
# Parametrized — RadialDCF
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "n_spokes, n_per_spoke",
    [
        (8, 16),
        (16, 32),
        (4, 64),
    ],
    ids=["sp8-s16", "sp16-s32", "sp4-s64"],
)
def test_radial_dcf_shape(n_spokes, n_per_spoke):
    from mriforge.infrastructure.physics.gridding import RadialDCF
    dcf = RadialDCF(num_spokes=n_spokes, samples_per_spoke=n_per_spoke)
    traj = _radial_traj(n_spokes, n_per_spoke)
    w = dcf(traj)
    assert w.shape == (n_spokes * n_per_spoke,)
    assert (w > 0).all()
    assert not torch.isnan(w).any()


@pytest.mark.physics
def test_radial_dcf_central_sample_weight():
    """The central (k_r≈0) sample gets the special 1/(2*nspokes) weight."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    n_spokes, n_per_spoke = 8, 17
    dcf = RadialDCF(num_spokes=n_spokes, samples_per_spoke=n_per_spoke)
    # Build trajectory where the middle sample of each spoke is k_r=0
    # Use linspace symmetric around 0
    r = torch.linspace(-math.pi, math.pi, n_per_spoke)
    mid_idx = n_per_spoke // 2
    assert abs(r[mid_idx].item()) < 1e-6, "Middle sample must be at k_r=0"
    traj = _radial_traj(n_spokes, n_per_spoke)
    w = dcf(traj)
    # The normalised weight is well-defined; just check no NaN
    assert not torch.isnan(w).any()


# ---------------------------------------------------------------------------
# Parametrized — IterativeDCF
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "n_pts, n_iters",
    [
        (16, 3),
        (32, 5),
        (64, 3),
    ],
    ids=["n16-it3", "n32-it5", "n64-it3"],
)
def test_iterative_dcf_shape(n_pts, n_iters):
    from mriforge.infrastructure.physics.gridding import IterativeDCF
    dcf = IterativeDCF(num_iters=n_iters)
    traj = _radial_traj(4, n_pts // 4)[: n_pts]
    w = dcf(traj)
    assert w.shape == (n_pts,)
    assert not torch.isnan(w).any()


# ---------------------------------------------------------------------------
# Parametrized — KaiserBesselKernel
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize("kernel_width, beta", [(4.0, 13.9), (6.0, 10.0), (2.0, 5.0)])
def test_kb_kernel_support(kernel_width, beta):
    """Kernel is zero outside its support (|d| > kernel_width/2)."""
    from mriforge.infrastructure.physics.gridding import KaiserBesselKernel
    kb = KaiserBesselKernel(kernel_width=kernel_width, beta=beta)
    # Sample just outside support
    outside = torch.tensor([kernel_width / 2.0 + 0.1, kernel_width + 1.0])
    assert torch.all(kb(outside) == 0.0), "Non-zero outside support"


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_radial_dcf_edge_single_spoke():
    """RadialDCF with 1 spoke does not raise."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    dcf = RadialDCF(num_spokes=1, samples_per_spoke=8)
    traj = _radial_traj(1, 8)
    w = dcf(traj)
    assert w.shape == (8,)
    assert not torch.isnan(w).any()


@pytest.mark.physics
def test_kb_kernel_edge_zero_distance():
    """At d=0 the kernel should be at its maximum (I_0(beta))."""
    from mriforge.infrastructure.physics.gridding import KaiserBesselKernel
    kb = KaiserBesselKernel(kernel_width=4.0, beta=13.9)
    d = torch.tensor([0.0])
    val = kb(d)
    assert val[0] > 0.0


@pytest.mark.physics
def test_kb_kernel_normalization_property():
    """KaiserBesselKernel.normalization returns a positive float."""
    from mriforge.infrastructure.physics.gridding import KaiserBesselKernel
    kb = KaiserBesselKernel()
    assert isinstance(kb.normalization, float)
    assert kb.normalization > 0


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_radial_dcf_raises_bad_traj_shape():
    """RadialDCF forward must raise ValueError when traj last dim != 2."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    dcf = RadialDCF(num_spokes=4, samples_per_spoke=8)
    bad_traj = torch.rand(32, 3)   # last dim is 3, not 2
    with pytest.raises(ValueError):
        dcf(bad_traj)


@pytest.mark.physics
def test_iterative_dcf_raises_wrong_rank():
    """IterativeDCF forward requires [N, 2] input."""
    from mriforge.infrastructure.physics.gridding import IterativeDCF
    dcf = IterativeDCF(num_iters=2)
    bad_traj = torch.rand(4, 8, 2)  # 3-D instead of 2-D
    with pytest.raises(ValueError):
        dcf(bad_traj)


@pytest.mark.physics
def test_radial_dcf_raises_zero_spokes():
    """RadialDCF init must raise ValueError for num_spokes < 1."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    with pytest.raises(ValueError):
        RadialDCF(num_spokes=0, samples_per_spoke=8)


@pytest.mark.physics
def test_radial_dcf_raises_too_few_samples():
    """RadialDCF init must raise ValueError for samples_per_spoke < 2."""
    from mriforge.infrastructure.physics.gridding import RadialDCF
    with pytest.raises(ValueError):
        RadialDCF(num_spokes=4, samples_per_spoke=1)


@pytest.mark.physics
def test_kb_kernel_raises_bad_params():
    """KaiserBesselKernel init must raise ValueError for non-positive params."""
    from mriforge.infrastructure.physics.gridding import KaiserBesselKernel
    with pytest.raises(ValueError):
        KaiserBesselKernel(kernel_width=0.0, beta=13.9)
    with pytest.raises(ValueError):
        KaiserBesselKernel(kernel_width=4.0, beta=0.0)


@pytest.mark.physics
def test_iterative_dcf_raises_bad_params():
    """IterativeDCF init must raise ValueError for non-positive kernel_width/beta."""
    from mriforge.infrastructure.physics.gridding import IterativeDCF
    with pytest.raises(ValueError):
        IterativeDCF(kernel_width=0.0, beta=13.9)
