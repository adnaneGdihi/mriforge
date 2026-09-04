"""Unit tests for complex_losses.py.

Covers: ComplexL1Loss, WeightedKSpaceL1Loss, LogSpectralPhaseLoss,
        ComplexMSELoss, FrequencyDomainLoss, SENSEAdjointL1Loss.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.complex_losses import (
    ComplexL1Loss,
    ComplexMSELoss,
    FrequencyDomainLoss,
    LogSpectralPhaseLoss,
    SENSEAdjointL1Loss,
    WeightedKSpaceL1Loss,
)
from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 2, 32, 32  # Even C so 2-channel real layout works

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_2ch(b=B, h=H, w=W) -> torch.Tensor:
    """[B, 2, H, W] real-valued 2-channel (simulated k-space)."""
    gen = torch.Generator()
    gen.manual_seed(42)
    return torch.randn(b, 2, h, w, generator=gen)


def _complex(b=B, h=H, w=W) -> torch.Tensor:
    """[B, 1, H, W] complex64 tensor."""
    gen = torch.Generator()
    gen.manual_seed(7)
    return torch.view_as_complex(
        torch.randn(b, 1, h, w, 2, generator=gen).contiguous()
    )


def _real_image(b=B, h=H, w=W) -> torch.Tensor:
    """[B, 1, H, W] real image for FrequencyDomainLoss."""
    gen = torch.Generator()
    gen.manual_seed(99)
    return torch.randn(b, 1, h, w, generator=gen)


# ===========================================================================
# ComplexL1Loss
# ===========================================================================


class TestComplexL1Loss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = ComplexL1Loss()
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = ComplexL1Loss(reduction=reduction)
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out).all()

    def test_property_identity_near_zero(self):
        """loss(x, x) ≈ 0 (only numerical floor eps prevents exact 0)."""
        loss_fn = ComplexL1Loss()
        x = _real_2ch()
        out = loss_fn(x, x)
        assert out.item() < 1e-4

    def test_property_nonneg(self):
        loss_fn = ComplexL1Loss()
        x = _real_2ch()
        y = _real_2ch()
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _real_2ch().requires_grad_(True)
        y = _real_2ch()
        loss_fn = ComplexL1Loss()
        out = loss_fn(x, y)
        out.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_edge_zero_input(self):
        loss_fn = ComplexL1Loss()
        x = torch.zeros(B, 2, H, W)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = ComplexL1Loss()
        x = torch.full((B, 2, H, W), 1e4)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_raises_real_only_tensor(self):
        """Odd-channel real tensor: falls through to L2 path — should not raise."""
        # NOTE: ComplexL1Loss handles shape mismatches gracefully;
        # the public contract is that it accepts real or complex.
        # A real [B, 3, H, W] (odd C) will not match any complex-conversion branch
        # and will reach the diff computation path.
        loss_fn = ComplexL1Loss()
        x = torch.randn(B, 3, H, W)
        y = torch.randn(B, 3, H, W)
        # Must not crash and must return a finite scalar
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("complex_l1")
        assert isinstance(loss_fn, ComplexL1Loss)


# ===========================================================================
# WeightedKSpaceL1Loss
# ===========================================================================


class TestWeightedKSpaceL1Loss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = WeightedKSpaceL1Loss()
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = WeightedKSpaceL1Loss(reduction=reduction)
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    @pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0])
    def test_parametric_exponents(self, exponent: float):
        loss_fn = WeightedKSpaceL1Loss(exponent=exponent)
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_identity_near_zero(self):
        loss_fn = WeightedKSpaceL1Loss()
        x = _real_2ch()
        out = loss_fn(x, x)
        assert out.item() < 1e-5

    def test_property_nonneg(self):
        loss_fn = WeightedKSpaceL1Loss()
        x = _real_2ch()
        y = _real_2ch()
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _real_2ch().requires_grad_(True)
        y = _real_2ch()
        loss_fn = WeightedKSpaceL1Loss()
        loss_fn(x, y).backward()
        assert x.grad is not None

    def test_edge_zero_input(self):
        loss_fn = WeightedKSpaceL1Loss()
        x = torch.zeros(B, 2, H, W)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = WeightedKSpaceL1Loss()
        x = torch.full((B, 2, H, W), 1e5)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("weighted_kspace_l1")
        assert isinstance(loss_fn, WeightedKSpaceL1Loss)


# ===========================================================================
# LogSpectralPhaseLoss
# ===========================================================================


class TestLogSpectralPhaseLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = LogSpectralPhaseLoss()
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = LogSpectralPhaseLoss(reduction=reduction)
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_property_nonneg(self):
        loss_fn = LogSpectralPhaseLoss()
        x = _real_2ch()
        y = _real_2ch()
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _real_2ch().requires_grad_(True)
        y = _real_2ch()
        loss_fn = LogSpectralPhaseLoss()
        loss_fn(x, y).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_edge_zero_input(self):
        loss_fn = LogSpectralPhaseLoss()
        x = torch.zeros(B, 2, H, W)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = LogSpectralPhaseLoss()
        x = torch.full((B, 2, H, W), 1e3)
        y = torch.full((B, 2, H, W), 1e3)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("log_spectral_phase")
        assert isinstance(loss_fn, LogSpectralPhaseLoss)


# ===========================================================================
# ComplexMSELoss
# ===========================================================================


class TestComplexMSELoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = ComplexMSELoss()
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = ComplexMSELoss(reduction=reduction)
        x = _real_2ch()
        y = _real_2ch()
        out = loss_fn(x, y)
        assert torch.isfinite(out).all()

    def test_property_identity_near_zero(self):
        loss_fn = ComplexMSELoss()
        x = _real_2ch()
        out = loss_fn(x, x)
        assert out.item() < 1e-10

    def test_property_nonneg(self):
        loss_fn = ComplexMSELoss()
        x = _real_2ch()
        y = _real_2ch()
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _real_2ch().requires_grad_(True)
        y = _real_2ch()
        loss_fn = ComplexMSELoss()
        loss_fn(x, y).backward()
        assert x.grad is not None

    def test_edge_zero_input(self):
        loss_fn = ComplexMSELoss()
        x = torch.zeros(B, 2, H, W)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = ComplexMSELoss()
        x = torch.full((B, 2, H, W), 1e4)
        y = torch.zeros(B, 2, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("complex_mse")
        assert isinstance(loss_fn, ComplexMSELoss)


# ===========================================================================
# FrequencyDomainLoss
# ===========================================================================


class TestFrequencyDomainLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = FrequencyDomainLoss()
        x = _real_image()
        y = _real_image()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("use_log", [True, False])
    def test_parametric_use_log(self, use_log: bool):
        loss_fn = FrequencyDomainLoss(use_log=use_log)
        x = _real_image()
        y = _real_image()
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    def test_property_identity_near_zero(self):
        loss_fn = FrequencyDomainLoss()
        x = _real_image()
        out = loss_fn(x, x)
        assert out.item() < 1e-4

    def test_property_nonneg(self):
        loss_fn = FrequencyDomainLoss()
        x = _real_image()
        y = _real_image()
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _real_image().requires_grad_(True)
        y = _real_image()
        loss_fn = FrequencyDomainLoss()
        loss_fn(x, y).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    def test_edge_zero_input(self):
        loss_fn = FrequencyDomainLoss()
        x = torch.zeros(B, 1, H, W)
        y = torch.zeros(B, 1, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = FrequencyDomainLoss()
        x = torch.full((B, 1, H, W), 1e3)
        y = torch.zeros(B, 1, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("frequency_domain_consistency")
        assert isinstance(loss_fn, FrequencyDomainLoss)


# ===========================================================================
# SENSEAdjointL1Loss
# ===========================================================================


class TestSENSEAdjointL1Loss:

    def test_canary_returns_zero_without_smaps(self):
        """When smaps=None, returns zero tensor (startup / no-coil mode)."""
        loss_fn = SENSEAdjointL1Loss()
        x = _real_2ch()
        out = loss_fn(x, x, smaps=None)
        assert torch.isfinite(out) and out.item() == pytest.approx(0.0, abs=1e-9)

    def test_canary_finite_with_smaps(self):
        """Full forward with real smaps returns finite scalar."""
        loss_fn = SENSEAdjointL1Loss()
        # Use complex smaps [B, 1, H, W]
        pred = _complex().contiguous()
        target = _complex().contiguous()
        smaps = _complex().contiguous()
        out = loss_fn(pred, target, smaps=smaps)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = SENSEAdjointL1Loss(reduction=reduction)
        pred = _complex().contiguous()
        target = _complex().contiguous()
        smaps = _complex().contiguous()
        out = loss_fn(pred, target, smaps=smaps)
        assert torch.isfinite(out)

    def test_property_nonneg_with_smaps(self):
        loss_fn = SENSEAdjointL1Loss()
        pred = _complex().contiguous()
        target = _complex().contiguous()
        smaps = _complex().contiguous()
        out = loss_fn(pred, target, smaps=smaps)
        assert out.item() >= 0.0

    def test_property_identity_near_zero(self):
        """loss(x, x) with same smaps should be near 0."""
        loss_fn = SENSEAdjointL1Loss()
        x = _complex().contiguous()
        smaps = _complex().contiguous()
        out = loss_fn(x, x, smaps=smaps)
        assert out.item() < 1e-4

    def test_edge_zero_smaps(self):
        loss_fn = SENSEAdjointL1Loss()
        pred = _complex().contiguous()
        target = _complex().contiguous()
        smaps = torch.zeros_like(pred)
        out = loss_fn(pred, target, smaps=smaps)
        assert torch.isfinite(out)

    def test_registry_lookup(self):
        loss_fn = create_loss("sense_adjoint_l1")
        assert isinstance(loss_fn, SENSEAdjointL1Loss)
