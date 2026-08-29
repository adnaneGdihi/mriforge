"""Canary / parametrized / edge-case / expected-failure tests for coil_compression.py.

Covers:
  - estimate_compression_matrix
  - apply_compression_matrix
  - SVDCoilCompressor
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_KSPACE, shape_id


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _kspace(b: int, c: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    real = torch.randn(b, c, h, w, generator=g)
    imag = torch.randn(b, c, h, w, generator=g)
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_coil_compression_imports():
    """All public symbols import cleanly."""
    from mriforge.infrastructure.physics.coil_compression import (
        SVDCoilCompressor,
        estimate_compression_matrix,
        apply_compression_matrix,
    )
    assert all([SVDCoilCompressor, estimate_compression_matrix, apply_compression_matrix])


@pytest.mark.physics
def test_canary_estimate_compression_matrix():
    """estimate_compression_matrix returns [V, C] complex matrix."""
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(1, 8, 32, 32)
    mat = estimate_compression_matrix(ks, num_virtual_coils=4)
    assert mat.shape == (4, 8)
    assert torch.is_complex(mat)


@pytest.mark.physics
def test_canary_apply_compression_matrix():
    """apply_compression_matrix returns [B, V, H, W] complex output."""
    from mriforge.infrastructure.physics.coil_compression import (
        estimate_compression_matrix,
        apply_compression_matrix,
    )
    ks = _kspace(1, 8, 32, 32)
    mat = estimate_compression_matrix(ks, num_virtual_coils=4)
    out = apply_compression_matrix(ks, mat)
    assert out.shape == (1, 4, 32, 32)
    assert torch.is_complex(out)


@pytest.mark.physics
def test_canary_svd_compressor():
    """SVDCoilCompressor forward runs and reduces coil count."""
    from mriforge.infrastructure.physics.coil_compression import SVDCoilCompressor
    ks = _kspace(1, 8, 32, 32)
    compressor = SVDCoilCompressor(num_virtual_coils=4, acs_size=8)
    out = compressor(ks)
    assert out.shape == (1, 4, 32, 32)


# ---------------------------------------------------------------------------
# Parametrized — estimate_compression_matrix
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w, v",
    [
        (1, 4, 32, 32, 2),
        (1, 8, 64, 64, 4),
        (2, 16, 32, 32, 8),
        (1, 8, 32, 64, 4),   # rectangular
    ],
    ids=["b1-c4-v2", "b1-c8-v4", "b2-c16-v8", "b1-c8-rect"],
)
def test_estimate_matrix_shape(b, c, h, w, v):
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(b, c, h, w)
    mat = estimate_compression_matrix(ks, num_virtual_coils=v)
    assert mat.shape == (v, c)


@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w, v",
    [
        (1, 4, 32, 32, 2),
        (1, 8, 64, 64, 4),
        (2, 16, 32, 32, 8),
    ],
    ids=["b1-c4-v2", "b1-c8-v4", "b2-c16-v8"],
)
def test_apply_matrix_output_shape(b, c, h, w, v):
    from mriforge.infrastructure.physics.coil_compression import (
        estimate_compression_matrix,
        apply_compression_matrix,
    )
    ks = _kspace(b, c, h, w)
    mat = estimate_compression_matrix(ks, num_virtual_coils=v)
    out = apply_compression_matrix(ks, mat)
    assert out.shape == (b, v, h, w)


# ---------------------------------------------------------------------------
# Parametrized — SVDCoilCompressor shape matrix
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    SHAPE_MATRIX_KSPACE,
    ids=[shape_id(s) for s in SHAPE_MATRIX_KSPACE],
)
def test_svd_compressor_shape_matrix(b, c, h, w):
    """SVDCoilCompressor reduces to half the coils for all kspace shapes."""
    from mriforge.infrastructure.physics.coil_compression import SVDCoilCompressor
    v = max(1, c // 2)
    ks = _kspace(b, c, h, w)
    compressor = SVDCoilCompressor(num_virtual_coils=v, acs_size=min(8, h, w))
    out = compressor(ks)
    assert out.shape == (b, v, h, w)
    assert not torch.isnan(out.real).any()


# ---------------------------------------------------------------------------
# ACS crop
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_estimate_with_acs_crop():
    """ACS-cropped calibration reduces to a smaller window without error."""
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(1, 8, 64, 64)
    mat = estimate_compression_matrix(ks, num_virtual_coils=4, acs_size=16)
    assert mat.shape == (4, 8)


# ---------------------------------------------------------------------------
# Shared-matrix mode
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_svd_compressor_shared_matrix():
    """SVDCoilCompressor with per_batch=False shares one matrix."""
    from mriforge.infrastructure.physics.coil_compression import SVDCoilCompressor
    ks = _kspace(2, 8, 32, 32)
    compressor = SVDCoilCompressor(num_virtual_coils=4, acs_size=8, per_batch=False)
    out = compressor(ks)
    assert out.shape == (2, 4, 32, 32)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_estimate_matrix_v_equals_c():
    """num_virtual_coils == C is the identity limit; matrix shape is [C, C]."""
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(1, 4, 16, 16)
    mat = estimate_compression_matrix(ks, num_virtual_coils=4)
    assert mat.shape == (4, 4)


@pytest.mark.physics
def test_estimate_matrix_single_virtual_coil():
    """Compressing to 1 virtual coil returns [1, C]."""
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(1, 8, 16, 16)
    mat = estimate_compression_matrix(ks, num_virtual_coils=1)
    assert mat.shape == (1, 8)


@pytest.mark.physics
def test_real_interleaved_input():
    """Real interleaved [B, 2*C, H, W] input is handled correctly."""
    from mriforge.infrastructure.physics.coil_compression import (
        estimate_compression_matrix,
        apply_compression_matrix,
    )
    # Real interleaved: 2*C=8 → C=4 coils
    ks_real = torch.randn(1, 8, 16, 16)   # float32, even channels
    mat = estimate_compression_matrix(ks_real, num_virtual_coils=2)
    assert mat.shape == (2, 4)
    out = apply_compression_matrix(ks_real, mat)
    # Output should be real interleaved: [B, 2*V, H, W]
    assert out.shape == (1, 4, 16, 16)


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_estimate_raises_too_many_virtual_coils():
    """num_virtual_coils > C must raise ValueError."""
    from mriforge.infrastructure.physics.coil_compression import estimate_compression_matrix
    ks = _kspace(1, 4, 16, 16)
    with pytest.raises(ValueError, match="num_virtual_coils"):
        estimate_compression_matrix(ks, num_virtual_coils=8)


@pytest.mark.physics
def test_apply_matrix_raises_coil_mismatch():
    """apply_compression_matrix raises ValueError when matrix C != kspace C."""
    from mriforge.infrastructure.physics.coil_compression import apply_compression_matrix
    ks = _kspace(1, 8, 16, 16)
    mat = torch.randn(4, 4, dtype=torch.complex64)   # expects C=4, got C=8
    with pytest.raises(ValueError):
        apply_compression_matrix(ks, mat)


@pytest.mark.physics
def test_svd_compressor_raises_bad_num_virtual():
    """SVDCoilCompressor raises ValueError for num_virtual_coils < 1."""
    from mriforge.infrastructure.physics.coil_compression import SVDCoilCompressor
    with pytest.raises(ValueError, match="num_virtual_coils"):
        SVDCoilCompressor(num_virtual_coils=0)


@pytest.mark.physics
def test_svd_compressor_raises_bad_acs_size():
    """SVDCoilCompressor raises ValueError for acs_size < 4."""
    from mriforge.infrastructure.physics.coil_compression import SVDCoilCompressor
    with pytest.raises(ValueError, match="acs_size"):
        SVDCoilCompressor(num_virtual_coils=4, acs_size=2)
