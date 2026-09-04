"""Tests for ``FrdMetric`` (Fréchet Radiomic Distance / FID).

Targets ``spectramr.core.metrics.frd``. FRD computes the Fréchet distance
between two Gaussian-modelled feature distributions; standard FID
formula. Verifies the identity property (real == fake → distance 0),
mean-shift behaviour, and the matrix-square-root fallback for singular
covariances.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectramr.core.metrics.frd import FrdMetric


# ---------------------------------------------------------------------------
# Identity / zero distance
# ---------------------------------------------------------------------------


def test_identical_distributions_yield_zero_distance() -> None:
    """Same feature matrix as both args → distance ≈ 0."""
    metric = FrdMetric()
    feats = torch.randn(50, 64)
    out = metric(feats, feats)
    # FID can be tiny but non-negative due to sqrtm round-off.
    assert out >= -1e-6
    assert out < 1e-3


def test_disjoint_distributions_yield_positive_distance() -> None:
    """Two clearly different distributions give a positive distance."""
    torch.manual_seed(0)
    metric = FrdMetric()
    real = torch.randn(50, 8) + 0.0
    fake = torch.randn(50, 8) + 5.0  # mean shift of 5
    out = metric(real, fake)
    assert out > 0


def test_mean_shift_dominates_distance() -> None:
    """A pure mean shift contributes ≈ ‖μ1 - μ2‖² to FRD.

    With identical covariances and mean shift δ, FRD = ‖δ‖².
    """
    torch.manual_seed(1)
    metric = FrdMetric()
    real = torch.randn(200, 4)
    fake = real.clone() + 1.0  # constant offset
    out = metric(real, fake)
    expected = 4.0  # ‖[1,1,1,1]‖² = 4
    # Tolerance: covariance match is approximate at finite N.
    assert abs(out - expected) < 0.5, f"got {out:.3f}, expected ≈ {expected}"


# ---------------------------------------------------------------------------
# Type / shape contract
# ---------------------------------------------------------------------------


def test_output_is_python_float() -> None:
    """Output is a Python float (suitable for JSON / dict serialisation)."""
    metric = FrdMetric()
    out = metric(torch.randn(20, 4), torch.randn(20, 4))
    assert isinstance(out, float)


def test_output_is_finite() -> None:
    """Output is finite even with high-dimensional features."""
    torch.manual_seed(2)
    metric = FrdMetric()
    out = metric(torch.randn(40, 32), torch.randn(40, 32))
    assert np.isfinite(out)


def test_accepts_different_sample_counts() -> None:
    """N_real and N_fake can differ — only feature dim D must match."""
    metric = FrdMetric()
    real = torch.randn(30, 8)
    fake = torch.randn(50, 8)
    out = metric(real, fake)
    assert np.isfinite(out)


# ---------------------------------------------------------------------------
# Edge / singular covariance
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_singular_covariance_falls_back_to_offset_path() -> None:
    """Constant features → singular Σ; the eps-offset branch keeps result finite.

    ``scipy.linalg.sqrtm`` emits a ``LinAlgWarning`` for genuinely
    singular inputs — that's exactly the branch we want to exercise
    here, so the filterwarnings is scoped to this test only.
    """
    metric = FrdMetric()
    real = torch.ones(10, 4)  # constant — singular covariance
    fake = torch.ones(10, 4) * 2.0
    out = metric(real, fake)
    assert np.isfinite(out)


@pytest.mark.parametrize(
    "n,d",
    [(20, 8), (50, 16), (100, 32)],
    ids=lambda v: f"{v}",
)
def test_shape_matrix(n: int, d: int) -> None:
    """Distance is finite across various (N, D) feature-matrix sizes."""
    torch.manual_seed(0)
    metric = FrdMetric()
    out = metric(torch.randn(n, d), torch.randn(n, d))
    assert np.isfinite(out)
