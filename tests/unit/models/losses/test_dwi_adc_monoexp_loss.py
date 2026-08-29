"""Tests for the DWI monoexponential ADC loss (audit 2026-07 I2).

Deterministic synthetic diffusion signals — no cluster data. The loss is the
first real consumer of the ``b_values`` batch key (previously an inert facade,
finding F3).
"""
from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.losses.dwi_adc_monoexp_loss import (
    DWIADCMonoexpLoss,
    fit_adc_loglinear,
)


def _monoexp(s0: float, adc: float, b: torch.Tensor, H: int = 4, W: int = 4):
    """S(b) = S0·exp(-b·ADC) as a [1, n_b, H, W] stack."""
    sig = s0 * torch.exp(-b * adc)  # [n_b]
    return sig.view(1, -1, 1, 1).expand(1, b.numel(), H, W).clone()


def test_fit_recovers_known_adc_and_s0() -> None:
    b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])
    s0, adc = 2.0, 1.2e-3
    images = _monoexp(s0, adc, b)
    adc_hat, log_s0_hat = fit_adc_loglinear(images, b)
    assert torch.allclose(adc_hat, torch.full_like(adc_hat, adc), atol=1e-6)
    assert torch.allclose(log_s0_hat, torch.full_like(log_s0_hat, math.log(s0)), atol=1e-5)


def test_loss_zero_for_perfect_monoexp() -> None:
    b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])
    images = _monoexp(2.0, 1.0e-3, b)
    loss = DWIADCMonoexpLoss(weight=1.0)
    assert loss(images, b).item() < 1e-8


def test_loss_positive_for_non_monoexp() -> None:
    b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])
    images = _monoexp(2.0, 1.0e-3, b)
    images[:, 2] = images[:, 2] * 1.5  # break the monoexponential decay
    loss = DWIADCMonoexpLoss(weight=1.0)
    assert loss(images, b).item() > 1e-4


def test_gradient_descends_toward_monoexp() -> None:
    b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])
    images = _monoexp(2.0, 1.0e-3, b)
    pred = (images * (1.0 + 0.2 * torch.randn_like(images))).clamp_min(0.05)
    pred = pred.detach().requires_grad_(True)
    loss = DWIADCMonoexpLoss(weight=1.0)(pred, b)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_lambda_adc_matches_reference() -> None:
    b = torch.tensor([0.0, 500.0, 1000.0, 1500.0])
    adc_true = 1.0e-3
    images = _monoexp(2.0, adc_true, b)
    ref = torch.full((1, 1, 4, 4), adc_true)
    loss = DWIADCMonoexpLoss(weight=1.0, lambda_adc=1.0)
    # Monoexp signal with matching reference -> near zero total loss.
    assert loss(images, b, adc_reference=ref).item() < 1e-8


def test_lambda_adc_without_reference_raises() -> None:
    b = torch.tensor([0.0, 1000.0])
    images = _monoexp(1.0, 1e-3, b)
    with pytest.raises(ValueError, match="adc_reference"):
        DWIADCMonoexpLoss(lambda_adc=1.0)(images, b)


def test_fewer_than_two_b_values_raises() -> None:
    with pytest.raises(ValueError, match=">= 2 b-values"):
        fit_adc_loglinear(torch.rand(1, 1, 4, 4), torch.tensor([1000.0]))


def test_degenerate_b_values_raise() -> None:
    with pytest.raises(ValueError, match="distinct"):
        fit_adc_loglinear(torch.rand(1, 2, 4, 4), torch.tensor([500.0, 500.0]))


def test_negative_weight_rejected() -> None:
    with pytest.raises(ValueError, match="weight"):
        DWIADCMonoexpLoss(weight=-1.0)


def test_registered_in_loss_registry() -> None:
    from mriforge.models.losses.registry import LossRegistry

    assert LossRegistry.is_registered("dwi_adc_monoexp")
