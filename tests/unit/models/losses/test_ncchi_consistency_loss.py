r"""Tests for the nc-χ (Bessel-ratio) consistency loss (SOTA plan T4).

Penalises deviation of the model's magnitude from the non-central-χ ML magnitude
of the noisy measurement (Bessel-ratio fixed point), generalising the moment-based
``RicianConsistencyLoss`` (L=1) to ``L`` coils. Corner case: high SNR ⇒ ML ≈ the
measurement (Gaussian consistency); ``L=1`` ⇒ Rician.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.ncchi_data_consistency import ncchi_ml_magnitude
from mriforge.models.losses.ncchi_consistency_loss import NcChiConsistencyLoss
from mriforge.models.losses.registry import create_loss


def test_registered_and_alias():
    assert isinstance(create_loss("ncchi_consistency", sigma=0.1), NcChiConsistencyLoss)


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        NcChiConsistencyLoss(sigma=0.0)
    with pytest.raises(ValueError):
        NcChiConsistencyLoss(sigma=0.1, n_coils=0)


def test_zero_when_prediction_equals_ncchi_ml_target():
    torch.manual_seed(0)
    sigma, L = 0.5, 4
    measurement = torch.rand(2, 1, 8, 8) * 5.0 + 1.0  # magnitudes
    target_ml = ncchi_ml_magnitude(measurement, sigma, n_coils=L, n_iter=30)
    loss = NcChiConsistencyLoss(sigma=sigma, n_coils=L)(prediction=target_ml, target=measurement)
    assert float(loss) < 1e-4


def test_nonzero_on_mismatch():
    sigma, L = 0.5, 4
    measurement = torch.rand(2, 1, 8, 8) * 5.0 + 1.0
    loss = NcChiConsistencyLoss(sigma=sigma, n_coils=L)(
        prediction=measurement + 2.0, target=measurement
    )
    assert float(loss) > 0.0


def test_high_snr_target_is_measurement():
    """At high SNR the nc-χ ML target ≈ the measurement (Gaussian consistency)."""
    sigma, L = 1e-3, 1
    measurement = torch.rand(1, 1, 8, 8) * 5.0 + 2.0
    # prediction == measurement should give ~0 loss when ML ≈ measurement.
    loss = NcChiConsistencyLoss(sigma=sigma, n_coils=L)(
        prediction=measurement, target=measurement
    )
    assert float(loss) < 1e-3


def test_handles_complex_and_interleaved():
    sigma = 0.3
    cpx = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
    out = NcChiConsistencyLoss(sigma=sigma)(prediction=cpx.abs(), target=cpx)
    assert out.ndim == 0 and torch.isfinite(out)
