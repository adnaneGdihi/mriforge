"""Tests for ``ClinicalTrustAnalyzer`` and the noise-floor estimator.

Targets ``spectramr.infrastructure.physics.clinical_trust_analyzer``.

Categories:

- ``estimate_noise_floor`` returns a non-negative scalar per batch
- ``estimate_noise_floor`` is robust to a few outliers (MAD)
- ``ClinicalTrustAnalyzer`` returns zero trust map when prediction is
  the iFFT of the raw k-space (no residual above noise)
- Trust map is non-zero when the prediction has bogus high-frequency
  content above the noise floor
- ``compute_null_space_residual`` returns a complex k-space residual
- Forward/adjoint operator overrides are respected
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.clinical_trust_analyzer import (
    ClinicalTrustAnalyzer,
    TrustAnalysisResult,
    estimate_noise_floor,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


# ---------------------------------------------------------------------------
# Noise-floor estimator
# ---------------------------------------------------------------------------


def test_noise_floor_non_negative() -> None:
    """``estimate_noise_floor`` is non-negative."""
    torch.manual_seed(0)
    k = torch.complex(torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16))
    sigma = estimate_noise_floor(k)
    assert (sigma >= 0).all()


def test_noise_floor_shape_is_per_batch() -> None:
    """Output shape is ``[B, 1, 1, 1]`` for broadcasting."""
    k = torch.complex(torch.randn(3, 1, 16, 16), torch.randn(3, 1, 16, 16))
    sigma = estimate_noise_floor(k)
    assert sigma.shape == (3, 1, 1, 1)


def test_noise_floor_zero_for_zero_kspace() -> None:
    """All-zero k-space → MAD = 0 → noise std = 0."""
    k = torch.zeros(1, 1, 16, 16, dtype=torch.complex64)
    sigma = estimate_noise_floor(k)
    assert sigma.item() == 0.0


def test_noise_floor_robust_to_outliers() -> None:
    """MAD-based estimator is bounded under a single huge outlier."""
    torch.manual_seed(0)
    k = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
    sigma_clean = estimate_noise_floor(k).item()
    # Inject a single very-large value in a corner.
    k[0, 0, 0, 0] = 1000.0 + 1000.0j
    sigma_outlier = estimate_noise_floor(k).item()
    # Allow some tolerance — but the MAD shouldn't blow up by orders of magnitude.
    assert sigma_outlier < 5.0 * sigma_clean


# ---------------------------------------------------------------------------
# ClinicalTrustAnalyzer — perfect-prediction case
# ---------------------------------------------------------------------------


def test_perfect_prediction_yields_zero_trust_map() -> None:
    """Prediction = iFFT(raw k-space) → null-space residual ≈ 0."""
    torch.manual_seed(0)
    H, W = 16, 16
    img = torch.complex(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
    raw_k = fft2c(img)
    analyzer = ClinicalTrustAnalyzer(noise_threshold_sigma=3.0)
    trust_map, _ = analyzer.compute_null_space_residual(img, raw_k)
    # Below the noise floor → relu zeros it out.
    assert trust_map.abs().max().item() < 1e-3


def test_kspace_residual_complex() -> None:
    """``kspace_residual`` is complex-valued."""
    torch.manual_seed(0)
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    raw_k = fft2c(img)
    analyzer = ClinicalTrustAnalyzer()
    _, k_res = analyzer.compute_null_space_residual(img, raw_k)
    assert torch.is_complex(k_res)


# ---------------------------------------------------------------------------
# Bogus-high-frequency case
# ---------------------------------------------------------------------------


def test_high_freq_hallucination_detected() -> None:
    """A prediction with injected high-frequency noise raises the trust map."""
    torch.manual_seed(0)
    H, W = 16, 16
    img = torch.complex(torch.randn(1, 1, H, W), torch.randn(1, 1, H, W))
    raw_k = fft2c(img) * 0.001  # very small noise floor
    # Bogus prediction: same shape but completely different content.
    bogus = torch.complex(
        torch.randn(1, 1, H, W) * 100.0,
        torch.randn(1, 1, H, W) * 100.0,
    )
    analyzer = ClinicalTrustAnalyzer(noise_threshold_sigma=1.0)
    trust_map, _ = analyzer.compute_null_space_residual(bogus, raw_k)
    # Some pixel in the trust map exceeds zero (residual above noise).
    assert trust_map.max().item() > 0.0


# ---------------------------------------------------------------------------
# Forward operator override
# ---------------------------------------------------------------------------


def test_custom_forward_operator_used() -> None:
    """A user-supplied forward operator replaces the default FFT."""

    captured: dict[str, torch.Tensor] = {}

    class _RecorderFwd(torch.nn.Module):
        def forward(self, x):
            captured["x"] = x
            return fft2c(x.to(torch.complex64))

    img = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
    raw_k = fft2c(img)
    analyzer = ClinicalTrustAnalyzer(forward_operator=_RecorderFwd())
    analyzer.compute_null_space_residual(img, raw_k)
    assert "x" in captured


# ---------------------------------------------------------------------------
# Mask path
# ---------------------------------------------------------------------------


def test_mask_zeroes_unsampled_residual() -> None:
    """Where mask = 0, residual contribution is zero (k-space is unmeasured)."""
    torch.manual_seed(0)
    img = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    raw_k = fft2c(img)
    bogus = torch.complex(
        torch.randn(1, 1, 8, 8) * 50.0, torch.randn(1, 1, 8, 8) * 50.0
    )
    mask = torch.zeros(1, 1, 8, 8)  # nothing measured
    analyzer = ClinicalTrustAnalyzer(use_adjoint=False)
    _, k_res = analyzer.compute_null_space_residual(bogus, raw_k, mask=mask)
    assert k_res.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# TrustAnalysisResult NamedTuple shape
# ---------------------------------------------------------------------------


def test_named_tuple_fields() -> None:
    """``TrustAnalysisResult`` is a NamedTuple with the documented fields."""
    fields = TrustAnalysisResult._fields
    assert "prediction" in fields
    assert "trust_map" in fields
    assert "jacobian_t1" in fields
    assert "jacobian_b0" in fields
    assert "kspace_residual" in fields
