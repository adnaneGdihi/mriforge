"""Canary / parametrized / edge-case / expected-failure tests for b0_mapping.py.

Covers:
  - phase_difference
  - estimate_b0_tikhonov
  - map_b0
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id
from tests.utils.phantoms import shepp_logan_2d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _complex_image(h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(1, 1, h, w, generator=g)
    im = torch.randn(1, 1, h, w, generator=g)
    return torch.complex(re, im)


def _uniform_image(h: int, w: int, phase: float = 0.0) -> torch.Tensor:
    """Return unit-magnitude complex image with uniform phase."""
    return torch.exp(1j * torch.full((1, 1, h, w), phase))


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_b0_mapping_imports():
    """Module imports cleanly."""
    from mriforge.infrastructure.physics.b0_mapping import (
        phase_difference,
        estimate_b0_tikhonov,
        map_b0,
    )
    assert all([phase_difference, estimate_b0_tikhonov, map_b0])


@pytest.mark.physics
def test_canary_phase_difference_trivial():
    """phase_difference on two trivial images returns [1, 1, H, W]."""
    from mriforge.infrastructure.physics.b0_mapping import phase_difference
    a = _complex_image(16, 16, seed=1)
    b = _complex_image(16, 16, seed=2)
    pd = phase_difference(a, b)
    assert pd.shape == (1, 1, 16, 16)
    assert not torch.isnan(pd).any()


@pytest.mark.physics
def test_canary_estimate_b0_tikhonov_trivial():
    """estimate_b0_tikhonov on a constant phase map runs cleanly."""
    from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov
    phase = torch.zeros(1, 1, 16, 16)
    b0 = estimate_b0_tikhonov(phase, t_shift=1e-4, max_cg_iterations=10)
    assert b0.shape == (1, 1, 16, 16)
    assert not torch.isnan(b0).any()


@pytest.mark.physics
def test_canary_map_b0_pipeline():
    """map_b0 (full pipeline) returns [1, 1, H, W] B0 estimate."""
    from mriforge.infrastructure.physics.b0_mapping import map_b0
    a = _complex_image(16, 16, seed=3)
    b = _complex_image(16, 16, seed=4)
    b0 = map_b0(a, b, t_shift=1e-4)
    assert b0.shape == (1, 1, 16, 16)
    assert not torch.isnan(b0).any()


# ---------------------------------------------------------------------------
# Parametrized — phase_difference
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "h, w",
    [(16, 16), (16, 32), (32, 32), (64, 64)],
    ids=["16x16", "16x32", "32x32", "64x64"],
)
def test_phase_difference_shape(h, w):
    from mriforge.infrastructure.physics.b0_mapping import phase_difference
    a = _complex_image(h, w, seed=5)
    b = _complex_image(h, w, seed=6)
    pd = phase_difference(a, b)
    assert pd.shape == (1, 1, h, w)


@pytest.mark.physics
def test_phase_difference_identical_inputs_near_zero():
    """Identical inputs should produce near-zero phase difference."""
    from mriforge.infrastructure.physics.b0_mapping import phase_difference
    img = _complex_image(16, 16, seed=7)
    pd = phase_difference(img, img)
    assert torch.allclose(pd, torch.zeros_like(pd), atol=1e-5)


@pytest.mark.physics
def test_phase_difference_range():
    """Output must be in [-pi, pi]."""
    from mriforge.infrastructure.physics.b0_mapping import phase_difference
    a = _complex_image(16, 16, seed=8)
    b = _complex_image(16, 16, seed=9)
    pd = phase_difference(a, b)
    assert (pd >= -math.pi - 1e-5).all()
    assert (pd <= math.pi + 1e-5).all()


# ---------------------------------------------------------------------------
# Parametrized — estimate_b0_tikhonov
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "h, w, t_shift",
    [
        (16, 16, 1e-4),
        (32, 32, 5e-4),
        (16, 32, 1e-4),   # rectangular
    ],
    ids=["16x16", "32x32", "16x32-rect"],
)
def test_estimate_b0_shape(h, w, t_shift):
    from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov
    phase = torch.randn(1, 1, h, w) * 0.1
    b0 = estimate_b0_tikhonov(phase, t_shift=t_shift, max_cg_iterations=10)
    assert b0.shape == (1, 1, h, w)
    assert not torch.isnan(b0).any()


@pytest.mark.physics
def test_estimate_b0_with_mask():
    """estimate_b0_tikhonov accepts and uses a binary mask."""
    from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov
    h, w = 16, 16
    phase = torch.randn(1, 1, h, w) * 0.1
    mask = torch.ones(1, 1, h, w)
    mask[:, :, :8, :] = 0.0   # mask out upper half
    b0 = estimate_b0_tikhonov(phase, t_shift=1e-4, mask=mask, max_cg_iterations=10)
    assert b0.shape == (1, 1, h, w)
    assert not torch.isnan(b0).any()


# ---------------------------------------------------------------------------
# Sanity-shape tests (SHAPE_MATRIX_2D, C=1)
# ---------------------------------------------------------------------------

@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, c, h, w",
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D if c == 1 and h <= 64],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D if c == 1 and h <= 64],
)
def test_map_b0_shape_matrix(b, c, h, w):
    """map_b0 output shape matches [1,1,H,W] for shape-matrix entries (B=1 forced)."""
    from mriforge.infrastructure.physics.b0_mapping import map_b0
    # map_b0 always returns [1, 1, H, W] (one B0 map per input)
    a = _complex_image(h, w, seed=10)
    bimg = _complex_image(h, w, seed=11)
    b0 = map_b0(a, bimg, t_shift=1e-4, max_cg_iterations=5)
    assert b0.shape == (1, 1, h, w)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_b0_edge_zero_phase_difference():
    """Zero phase difference → near-zero B0 estimate."""
    from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov
    phase = torch.zeros(1, 1, 16, 16)
    b0 = estimate_b0_tikhonov(phase, t_shift=1e-4, max_cg_iterations=20)
    # With zero RHS the CG solution should be zero
    assert torch.allclose(b0, torch.zeros_like(b0), atol=1e-4)


@pytest.mark.physics
def test_b0_edge_very_small_t_shift():
    """Very small t_shift (near 0) should not cause division-by-zero."""
    from mriforge.infrastructure.physics.b0_mapping import estimate_b0_tikhonov
    phase = torch.randn(1, 1, 8, 8) * 0.1
    b0 = estimate_b0_tikhonov(phase, t_shift=1e-10, max_cg_iterations=5)
    assert not torch.isnan(b0).any()


@pytest.mark.physics
def test_b0_edge_shepp_logan_phantom():
    """map_b0 with a Shepp-Logan phantom as magnitude doesn't produce NaN."""
    from mriforge.infrastructure.physics.b0_mapping import map_b0
    sl = shepp_logan_2d(32)
    # Build complex images from the phantom
    phase_a = torch.zeros(32, 32)
    phase_b = sl * 0.1 * math.pi
    img_a = (sl * torch.exp(1j * phase_a)).unsqueeze(0).unsqueeze(0)
    img_b = (sl * torch.exp(1j * phase_b)).unsqueeze(0).unsqueeze(0)
    b0 = map_b0(img_a, img_b, t_shift=1e-4, max_cg_iterations=5)
    assert not torch.isnan(b0).any()
    assert b0.shape == (1, 1, 32, 32)
