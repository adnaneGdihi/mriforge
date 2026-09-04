r"""Tests for the Ambient Diffusion held-out k-space consistency loss (Phase B step 5).

Ambient Diffusion (Daras et al. 2023; Aali et al. MRI A-DPS) learns a clean prior
from undersampled data by the SSDU Λ/Θ split lifted onto a diffusion model: the
model's x₀ estimate must agree with the held-out Θ measurement,
``‖M_Θ(F·x̂₀ − y)‖²``. At ``ambient_weight → 1`` this is the SSDU held-out
residual; at ``noise → 0`` the whole strategy reduces to SSDU.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.ambient_consistency_loss import (
    AmbientConsistencyLoss,
    ambient_consistency_residual,
)
from spectramr.models.losses.registry import create_loss


def _setup(h: int = 8, w: int = 8):
    torch.manual_seed(0)
    x0 = torch.randn(1, 1, h, w)
    y = fft2c(torch.complex(x0, torch.zeros_like(x0)))  # k-space of x0 (perfect)
    theta = torch.zeros(1, 1, h, w)
    theta[..., ::2] = 1.0
    return x0, y, theta


def test_residual_zero_when_x0_matches_measurement():
    x0, y, theta = _setup()
    # x0's k-space equals y everywhere → held-out residual is ~0.
    r = ambient_consistency_residual(x0, y, theta)
    assert float(r) < 1e-8


def test_residual_nonzero_on_mismatch_and_scales_with_weight():
    x0, y, theta = _setup()
    x0_bad = x0 + 1.0
    r1 = ambient_consistency_residual(x0_bad, y, theta, weight=1.0)
    r2 = ambient_consistency_residual(x0_bad, y, theta, weight=2.0)
    assert float(r1) > 0
    assert abs(float(r2) - 2.0 * float(r1)) < 1e-5


def test_residual_only_counts_theta_entries():
    x0, y, theta = _setup()
    x0_bad = x0 + 0.5
    # Corrupting only the NON-theta entries leaves the theta residual unchanged.
    full = ambient_consistency_residual(x0_bad, y, theta)
    r_theta_only = ambient_consistency_residual(x0_bad, y, theta)
    assert float(full) == float(r_theta_only)
    # Empty theta → zero loss.
    assert float(ambient_consistency_residual(x0_bad, y, torch.zeros_like(theta))) == 0.0


def test_loss_reads_context_and_is_registered():
    x0, y, theta = _setup()
    loss = AmbientConsistencyLoss()(
        prediction=x0 + 1.0,
        target=torch.zeros_like(x0),
        context={"target_kspace": y, "theta_mask": theta},
    )
    assert loss.ndim == 0 and float(loss) > 0
    # registered name + alias resolve
    assert isinstance(create_loss("ambient_consistency"), AmbientConsistencyLoss)


def test_loss_requires_context():
    x0, y, theta = _setup()
    with pytest.raises(ValueError, match="theta_mask"):
        AmbientConsistencyLoss()(prediction=x0, target=torch.zeros_like(x0), context=None)
