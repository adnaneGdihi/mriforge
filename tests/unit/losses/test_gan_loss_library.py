"""Unit tests for gan_loss_library.py — four archetypes per class.

Covers: StandardGANLoss, LSGANLoss, RALSGANLoss, WGANLoss, HingeLoss,
        R1RegularizationLoss, get_gan_loss (ValueError path).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.losses.gan_loss_library import (
    HingeLoss,
    LSGANLoss,
    RALSGANLoss,
    R1RegularizationLoss,
    StandardGANLoss,
    WGANLoss,
    get_gan_loss,
)
from spectramr.models.losses.registry import create_loss

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B = 4
_DISC_SHAPE = (B, 1)  # discriminator output shape


def _real_fake() -> tuple[torch.Tensor, torch.Tensor]:
    """Fake / real discriminator outputs: real ≈ +2, fake ≈ -2."""
    real = torch.full(_DISC_SHAPE, 2.0)
    fake = torch.full(_DISC_SHAPE, -2.0)
    return real, fake


# ===========================================================================
# StandardGANLoss (BCE)
# ===========================================================================


class TestStandardGANLoss:
    """Canary, parametric, property, edge, raises tests."""

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = StandardGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert torch.isfinite(g_loss)
        assert g_loss.shape == ()  # scalar

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("smoothing", [0.0, 0.1, 0.2])
    def test_parametric_label_smoothing(self, smoothing: float):
        loss_fn = StandardGANLoss(label_smoothing=smoothing)
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=real, fake_outputs_d=fake
        )
        assert torch.isfinite(r_loss) and torch.isfinite(f_loss)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_nonneg_generator_loss(self):
        """BCE G-loss = E[BCEWithLogits(fake, 1)] ≥ 0."""
        loss_fn = StandardGANLoss()
        _, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert g_loss.item() >= 0.0

    def test_property_gradient_reaches_input(self):
        fake = torch.randn(B, 1, requires_grad=True)
        loss_fn = StandardGANLoss()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        g_loss.backward()
        assert fake.grad is not None
        assert torch.isfinite(fake.grad).all()

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_zero_input(self):
        loss_fn = StandardGANLoss()
        zeros = torch.zeros(B, 1)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=zeros)
        assert torch.isfinite(g_loss)

    def test_edge_large_magnitude(self):
        loss_fn = StandardGANLoss()
        large = torch.full(_DISC_SHAPE, 1e6)
        # Large positive logits → gen loss near 0 (D already convinced)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=large)
        assert torch.isfinite(g_loss)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_unknown_via_registry(self):
        with pytest.raises((ValueError, RuntimeError, KeyError)):
            create_loss("gan_standard_nonexistent_xyzzy_42")


# ===========================================================================
# LSGANLoss
# ===========================================================================


class TestLSGANLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = LSGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert torch.isfinite(g_loss) and g_loss.shape == ()

    @pytest.mark.parametrize("smoothing", [0.0, 0.05])
    def test_parametric_label_smoothing(self, smoothing: float):
        loss_fn = LSGANLoss(label_smoothing=smoothing)
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(real_outputs_d=real, fake_outputs_d=fake)
        assert torch.isfinite(r_loss) and torch.isfinite(f_loss)

    def test_property_lsgan_loss_nonneg(self):
        """LSGAN loss is MSE, always ≥ 0."""
        loss_fn = LSGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert g_loss.item() >= 0.0

    def test_property_gradient_reaches_input(self):
        fake = torch.randn(B, 1, requires_grad=True)
        loss_fn = LSGANLoss()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        g_loss.backward()
        assert fake.grad is not None

    def test_edge_zero_input(self):
        loss_fn = LSGANLoss()
        zeros = torch.zeros(B, 1)
        # LSGAN: gen loss = mean((0 - 1)^2) = 1.0
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=zeros)
        assert torch.isfinite(g_loss)

    def test_edge_large_magnitude(self):
        loss_fn = LSGANLoss()
        large = torch.full(_DISC_SHAPE, 1e4)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=large)
        assert torch.isfinite(g_loss)


# ===========================================================================
# RALSGANLoss
# ===========================================================================


class TestRALSGANLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = RALSGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(
            fake_outputs_d=fake, real_outputs_d=real
        )
        assert torch.isfinite(g_loss) and g_loss.shape == ()

    @pytest.mark.parametrize("smoothing", [0.0, 0.1])
    def test_parametric_label_smoothing(self, smoothing: float):
        loss_fn = RALSGANLoss(label_smoothing=smoothing)
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=real, fake_outputs_d=fake
        )
        assert torch.isfinite(r_loss) and torch.isfinite(f_loss)

    def test_property_nonneg(self):
        """RALSGAN gen loss ≥ 0."""
        loss_fn = RALSGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(
            fake_outputs_d=fake, real_outputs_d=real
        )
        assert g_loss.item() >= 0.0

    def test_property_gradient_reaches_inputs(self):
        real = torch.randn(B, 1, requires_grad=True)
        fake = torch.randn(B, 1, requires_grad=True)
        loss_fn = RALSGANLoss()
        g_loss = loss_fn.compute_generator_loss(
            fake_outputs_d=fake, real_outputs_d=real
        )
        g_loss.backward()
        assert fake.grad is not None

    def test_edge_raises_without_real(self):
        """Generator loss requires real_outputs_d."""
        loss_fn = RALSGANLoss()
        _, fake = _real_fake()
        with pytest.raises(ValueError, match="real_outputs_d"):
            loss_fn.compute_generator_loss(fake_outputs_d=fake)

    def test_edge_zero_input(self):
        loss_fn = RALSGANLoss()
        zeros = torch.zeros(B, 1)
        ones = torch.zeros(B, 1)
        g_loss = loss_fn.compute_generator_loss(
            fake_outputs_d=zeros, real_outputs_d=ones
        )
        assert torch.isfinite(g_loss)


# ===========================================================================
# WGANLoss
# ===========================================================================


class TestWGANLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = WGANLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert torch.isfinite(g_loss) and g_loss.shape == ()

    @pytest.mark.parametrize("smoothing", [0.0, 0.05])
    def test_parametric_label_smoothing(self, smoothing: float):
        loss_fn = WGANLoss(label_smoothing=smoothing)
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=real, fake_outputs_d=fake
        )
        assert torch.isfinite(r_loss) and torch.isfinite(f_loss)

    def test_property_wgan_generator_equals_neg_mean(self):
        """WGAN gen loss = -mean(fake)."""
        loss_fn = WGANLoss()
        fake = torch.tensor([[2.0], [4.0]])
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert torch.allclose(g_loss, torch.tensor(-3.0))

    def test_property_gradient_reaches_input(self):
        fake = torch.randn(B, 1, requires_grad=True)
        loss_fn = WGANLoss()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        g_loss.backward()
        assert fake.grad is not None

    def test_edge_zero_input(self):
        loss_fn = WGANLoss()
        zeros = torch.zeros(B, 1)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=zeros)
        assert g_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_large_magnitude_finite(self):
        loss_fn = WGANLoss()
        large = torch.full(_DISC_SHAPE, 1e6)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=large)
        assert torch.isfinite(g_loss)


# ===========================================================================
# HingeLoss
# ===========================================================================


class TestHingeLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = HingeLoss()
        real, fake = _real_fake()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        assert torch.isfinite(g_loss) and g_loss.shape == ()

    @pytest.mark.parametrize("smoothing", [0.0, 0.1])
    def test_parametric_label_smoothing_disc(self, smoothing: float):
        loss_fn = HingeLoss(label_smoothing=smoothing)
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=real, fake_outputs_d=fake
        )
        assert torch.isfinite(r_loss) and torch.isfinite(f_loss)

    def test_property_disc_losses_nonneg(self):
        """Hinge D-losses are relu-based, always ≥ 0."""
        loss_fn = HingeLoss()
        real, fake = _real_fake()
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=real, fake_outputs_d=fake
        )
        assert r_loss.item() >= 0.0
        assert f_loss.item() >= 0.0

    def test_property_gradient_reaches_input(self):
        fake = torch.randn(B, 1, requires_grad=True)
        loss_fn = HingeLoss()
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=fake)
        g_loss.backward()
        assert fake.grad is not None

    def test_edge_hinge_clamp_at_margin(self):
        """For real outputs already above margin, D real loss should be 0."""
        loss_fn = HingeLoss()
        above_margin = torch.full(_DISC_SHAPE, 2.0)  # above 1.0 real target
        below_margin = torch.full(_DISC_SHAPE, 2.0)  # symmetrically: fake loss = 0
        r_loss, f_loss = loss_fn.compute_discriminator_loss(
            real_outputs_d=above_margin, fake_outputs_d=-below_margin
        )
        assert r_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_edge_zero_input(self):
        loss_fn = HingeLoss()
        zeros = torch.zeros(B, 1)
        g_loss = loss_fn.compute_generator_loss(fake_outputs_d=zeros)
        assert torch.isfinite(g_loss)


# ===========================================================================
# R1RegularizationLoss
# ===========================================================================


class TestR1RegularizationLoss:

    def test_canary_builds_and_skips_non_module_discriminator(self):
        """When discriminator is not an nn.Module, returns 0 (safety path)."""
        loss_fn = R1RegularizationLoss(weight=10.0)
        real = torch.randn(2, 1, 8, 8)
        # Passing a tensor as 'discriminator' triggers the safety branch
        out = loss_fn(torch.zeros(2, 1, 8, 8), real)
        assert torch.isfinite(out)

    def test_canary_real_discriminator_finite(self):
        """Full forward with a simple discriminator returns finite scalar."""
        loss_fn = R1RegularizationLoss(weight=5.0)

        class TinyDisc(nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3), keepdim=True)

        disc = TinyDisc()
        real = torch.randn(2, 1, 8, 8)
        out = loss_fn(disc, real)
        assert torch.isfinite(out) and out.shape == ()

    def test_property_nonneg(self):
        """R1 penalty = 0.5 * ||grad||^2 ≥ 0."""
        loss_fn = R1RegularizationLoss(weight=1.0)

        class TinyDisc(nn.Module):
            def forward(self, x):
                return x.sum(dim=(1, 2, 3), keepdim=True)

        disc = TinyDisc()
        real = torch.randn(2, 1, 8, 8)
        out = loss_fn(disc, real)
        assert out.item() >= 0.0

    def test_property_zero_weight_returns_zero(self):
        loss_fn = R1RegularizationLoss(weight=0.0)
        real = torch.randn(2, 1, 8, 8)
        out = loss_fn(nn.Identity(), real)
        assert out.item() == pytest.approx(0.0, abs=1e-9)

    def test_edge_large_gradient(self):
        """Large gradients should not produce NaN (nan-guard active)."""
        loss_fn = R1RegularizationLoss(weight=1.0)

        class ScaledDisc(nn.Module):
            def forward(self, x):
                return (x * 1e4).sum(dim=(1, 2, 3), keepdim=True)

        disc = ScaledDisc()
        real = torch.randn(2, 1, 8, 8)
        out = loss_fn(disc, real)
        assert torch.isfinite(out)

    def test_edge_single_sample(self):
        loss_fn = R1RegularizationLoss(weight=1.0)

        class TinyDisc(nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2, 3), keepdim=True)

        disc = TinyDisc()
        real = torch.randn(1, 1, 4, 4)
        out = loss_fn(disc, real)
        assert torch.isfinite(out)


# ===========================================================================
# get_gan_loss — registry / ValueError
# ===========================================================================


class TestGetGanLoss:

    @pytest.mark.parametrize("name", ["bce", "hinge", "lsgan", "wgan", "ralsgan"])
    def test_canary_known_names_return_callables(self, name: str):
        result = get_gan_loss(name)
        assert isinstance(result, tuple) and len(result) == 2
        d_fn, g_fn = result
        assert callable(d_fn) and callable(g_fn)

    def test_parametric_bce_forward_finite(self):
        d_fn, g_fn = get_gan_loss("bce")
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        d_loss = d_fn(real, fake)
        g_loss = g_fn(fake)
        assert torch.isfinite(d_loss) and torch.isfinite(g_loss)

    def test_property_wgan_d_loss_sign(self):
        """WGAN D-loss = -E[real] + E[fake]; perfect discriminator: real > fake."""
        d_fn, _ = get_gan_loss("wgan")
        real = torch.full((4, 1), 3.0)
        fake = torch.full((4, 1), -3.0)
        # D-loss should be negative (D winning; WGAN wants to maximise this)
        d_loss = d_fn(real, fake)
        assert d_loss.item() < 0.0

    def test_raises_unknown_name(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_gan_loss("nonexistent_gan_type_xyz")

    def test_edge_ralsgan_requires_real_for_g(self):
        """RALSGAN g_loss callable requires real_outputs argument."""
        _, g_fn = get_gan_loss("ralsgan")
        fake = torch.randn(4, 1)
        with pytest.raises((ValueError, TypeError)):
            # Calling without real_outputs should fail or behave per implementation
            g_fn(fake)
