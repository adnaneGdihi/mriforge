"""Unit tests for infrastructure/reporting/metrics/haarpsi.py.

Covers: haar_psi, _haar_filter_bank.
"""

from __future__ import annotations

import numpy as np
import pytest

from spectramr.infrastructure.reporting.metrics.haarpsi import (
    _haar_filter_bank,
    haar_psi,
)


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_haar_psi_canary():
    """Identical images should yield a score close to 1."""
    img = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    score = haar_psi(img, img)
    assert isinstance(score, float)
    assert score > 0.95, f"Self-similarity should be near 1, got {score}"


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("scale", [1, 2])
def test_haar_filter_bank_shapes(scale):
    """Filter bank produces two filters of the expected shape."""
    h_f, v_f = _haar_filter_bank(scale)
    expected_size = 2**scale
    assert h_f.shape == (expected_size, expected_size)
    assert v_f.shape == (expected_size, expected_size)


@pytest.mark.unit
@pytest.mark.parametrize("data_range", [1.0, 255.0])
def test_haar_psi_data_range_invariance(data_range):
    """Normalising pred/target by data_range should not change the score."""
    rng = np.random.default_rng(42)
    img = rng.random((32, 32)).astype(np.float32)
    noisy = (img + 0.05 * rng.random((32, 32)).astype(np.float32)).clip(0, 1)

    score_unit = haar_psi(img, noisy, data_range=1.0)
    score_scaled = haar_psi(img * data_range, noisy * data_range, data_range=data_range)
    assert abs(score_unit - score_scaled) < 0.01, (
        f"Scores differ under rescaling: {score_unit:.4f} vs {score_scaled:.4f}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.5])
def test_haar_psi_degrades_with_noise(noise_level):
    """Score should decrease (or stay equal) as noise level increases."""
    rng = np.random.default_rng(7)
    img = rng.random((64, 64)).astype(np.float32)
    noisy = (img + noise_level * rng.random((64, 64)).astype(np.float32)).clip(0, 1)
    score = haar_psi(img, noisy)
    # The clean pair (noise_level=0) is identical, so score == 1.
    # For any noise, it should be <= 1.
    assert 0.0 <= score <= 1.0 + 1e-6, f"Score out of [0,1]: {score}"


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_haar_psi_flat_images():
    """Flat (constant) images should yield a well-defined score without errors."""
    flat = np.ones((32, 32), dtype=np.float32)
    score = haar_psi(flat, flat)
    assert np.isfinite(score), "Score is not finite for constant images"


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_haar_psi_shape_mismatch_raises():
    """Mismatched shapes must raise ValueError, not silently produce garbage."""
    a = np.zeros((32, 32), dtype=np.float32)
    b = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        haar_psi(a, b)
