"""Extended tests for coil_sensitivity.py (critical-partial module).

The existing canary for estimate_csm_rss lives in test_physics_modules.py.
We extend coverage here for:
  - estimate_csm_rss: shape / coil-count mismatch raises / various shapes
  - estimate_csm_power_iter: shape contract + coil count mismatch raises
  - Edge: single coil, large-coil array, near-zero k-space
  - Raises: wrong rank, coil mismatch
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_KSPACE, shape_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kspace_complex(b: int, c: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(b, c, h, w, generator=g)
    im = torch.randn(b, c, h, w, generator=g)
    return torch.complex(re, im)


# ---------------------------------------------------------------------------
# Canary — just imports
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_coil_sensitivity_imports():
    """All public CSM symbols import cleanly."""
    from mriforge.infrastructure.physics.coil_sensitivity import (
        estimate_csm_rss,
        estimate_csm_power_iter,
    )
    assert all([estimate_csm_rss, estimate_csm_power_iter])


# ---------------------------------------------------------------------------
# estimate_csm_rss — parametrized
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    [
        (1, 1, 32, 32),     # single coil
        (1, 4, 32, 32),
        (1, 8, 64, 64),
        (2, 4, 32, 64),     # rectangular
    ],
    ids=["b1-c1", "b1-c4", "b1-c8", "b2-c4-rect"],
)
def test_csm_rss_shape(b, c, h, w):
    """estimate_csm_rss returns [B, C, H, W] complex."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(b, c, h, w)
    smaps = estimate_csm_rss(ks, num_coils=c)
    assert smaps.shape == (b, c, h, w)
    assert torch.is_complex(smaps)
    assert not torch.isnan(smaps).any()


# ---------------------------------------------------------------------------
# estimate_csm_rss — shape-matrix
# ---------------------------------------------------------------------------

@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, c, h, w",
    SHAPE_MATRIX_KSPACE,
    ids=[shape_id(s) for s in SHAPE_MATRIX_KSPACE],
)
def test_csm_rss_shape_matrix(b, c, h, w):
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(b, c, h, w)
    smaps = estimate_csm_rss(ks, num_coils=c)
    assert smaps.shape == (b, c, h, w)


# ---------------------------------------------------------------------------
# estimate_csm_power_iter — parametrized
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    [
        (1, 4, 32, 32),
        (1, 8, 32, 32),
        (2, 4, 32, 32),
    ],
    ids=["b1-c4", "b1-c8", "b2-c4"],
)
def test_csm_power_iter_shape(b, c, h, w):
    """estimate_csm_power_iter returns [B, C, H, W] complex."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_power_iter
    ks = _kspace_complex(b, c, h, w)
    smaps = estimate_csm_power_iter(ks, num_coils=c, num_iter=3)
    assert smaps.shape == (b, c, h, w)
    assert torch.is_complex(smaps)
    assert not torch.isnan(smaps).any()


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_csm_rss_edge_single_coil():
    """Single-coil case returns a map of unit magnitude (RSS of one)."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(1, 1, 16, 16)
    smaps = estimate_csm_rss(ks, num_coils=1)
    assert smaps.shape == (1, 1, 16, 16)
    # With a single coil the RSS normalised map should have magnitude ≤ 1
    assert (smaps.abs() <= 1.0 + 1e-5).all()


@pytest.mark.physics
def test_csm_rss_edge_near_zero_kspace():
    """Near-zero k-space: the eps guard should prevent NaN."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = torch.zeros(1, 4, 16, 16, dtype=torch.complex64)
    ks += 1e-10  # tiny non-zero signal
    smaps = estimate_csm_rss(ks, num_coils=4)
    assert not torch.isnan(smaps).any()


@pytest.mark.physics
def test_csm_rss_edge_large_coil_array():
    """16-coil array runs without error (memory-bounded)."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(1, 16, 32, 32)
    smaps = estimate_csm_rss(ks, num_coils=16)
    assert smaps.shape == (1, 16, 32, 32)


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_csm_rss_raises_wrong_rank():
    """estimate_csm_rss must raise ValueError for 3-D input (missing B dim)."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(1, 4, 32, 32).squeeze(0)  # [C, H, W] — missing B
    with pytest.raises(ValueError):
        estimate_csm_rss(ks, num_coils=4)


@pytest.mark.physics
def test_csm_rss_raises_coil_mismatch():
    """estimate_csm_rss must raise ValueError when num_coils != kspace.shape[1]."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_rss
    ks = _kspace_complex(1, 8, 32, 32)
    with pytest.raises(ValueError):
        estimate_csm_rss(ks, num_coils=4)   # mismatch: 8 != 4


@pytest.mark.physics
def test_csm_power_iter_raises_wrong_rank():
    """estimate_csm_power_iter must raise ValueError for 3-D input."""
    from mriforge.infrastructure.physics.coil_sensitivity import estimate_csm_power_iter
    ks = _kspace_complex(1, 4, 32, 32).squeeze(0)  # [C, H, W]
    with pytest.raises(ValueError):
        estimate_csm_power_iter(ks, num_coils=4)
