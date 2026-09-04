"""Unit tests for ssim_loss.py.

Covers: SSIMLoss, MSSSIMLoss, and the module-level convenience functions.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.ssim_loss import MSSSIMLoss, SSIMLoss, msssim_loss, ssim_loss
from spectramr.models.losses.registry import create_loss

B, C, H, W = 2, 1, 64, 64  # SSIM needs reasonably large images


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


# ===========================================================================
# SSIMLoss
# ===========================================================================


class TestSSIMLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = SSIMLoss()
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = SSIMLoss(reduction=reduction)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize("window_size", [7, 11])
    def test_parametric_window_sizes(self, window_size: int):
        loss_fn = SSIMLoss(window_size=window_size)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_identity_is_zero(self):
        """SSIM(x, x) = 1  →  loss = 1 - 1 = 0."""
        loss_fn = SSIMLoss()
        x = _img()
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-4)

    def test_property_nonneg(self):
        """Loss = 1 - SSIM ∈ [0, 1] for normalised images."""
        loss_fn = SSIMLoss()
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert 0.0 <= out.item() <= 1.01  # allow slight numerical margin

    def test_property_gradient_reaches_both_inputs(self):
        x = _img().requires_grad_(True)
        y = _img(seed=1)
        loss_fn = SSIMLoss()
        out = loss_fn(x, y)
        out.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_zero_input(self):
        """Zero-constant images should still return finite loss."""
        loss_fn = SSIMLoss()
        x = torch.zeros(B, C, H, W)
        y = torch.zeros(B, C, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        """Values >>1 should not overflow."""
        loss_fn = SSIMLoss(auto_range=True)
        x = torch.full((B, C, H, W), 1e3)
        y = torch.full((B, C, H, W), 1e3)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_multichannel(self):
        loss_fn = SSIMLoss()
        x = _img(c=3)
        y = _img(c=3, seed=2)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_even_window_size(self):
        with pytest.raises(ValueError, match="odd"):
            SSIMLoss(window_size=10)

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError):
            SSIMLoss(reduction="invalid_reduction_xyz")

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("ssim")
        assert isinstance(loss_fn, SSIMLoss)


# ===========================================================================
# MSSSIMLoss
# ===========================================================================


class TestMSSSIMLoss:

    def test_canary_builds_and_forward_finite(self):
        loss_fn = MSSSIMLoss()
        x = _img()
        y = _img(seed=3)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reductions(self, reduction: str):
        loss_fn = MSSSIMLoss(reduction=reduction)
        x = _img()
        y = _img(seed=3)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_property_identity_near_zero(self):
        """MS-SSIM(x,x) = 1, loss = 0."""
        loss_fn = MSSSIMLoss()
        x = _img()
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-3)

    def test_property_nonneg(self):
        loss_fn = MSSSIMLoss()
        x = _img()
        y = _img(seed=3)
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_input(self):
        x = _img().requires_grad_(True)
        y = _img(seed=3)
        loss_fn = MSSSIMLoss()
        loss_fn(x, y).backward()
        assert x.grad is not None

    def test_edge_minimum_scale(self):
        """Images just large enough for all 5 scales."""
        loss_fn = MSSSIMLoss()
        x = _img(h=32, w=32)
        y = _img(h=32, w=32, seed=4)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_zero_input(self):
        loss_fn = MSSSIMLoss()
        x = torch.zeros(B, C, H, W)
        y = torch.zeros(B, C, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_raises_shape_mismatch(self):
        loss_fn = MSSSIMLoss()
        x = _img()
        y = _img(h=32, w=32)
        with pytest.raises(ValueError, match="shape"):
            loss_fn(x, y)

    def test_registry_lookup(self):
        loss_fn = create_loss("ms_ssim")
        assert isinstance(loss_fn, MSSSIMLoss)

    def test_amp_fp16_offscale_is_finite(self):
        """AMP-fp16 regression: an off-scale prediction under an active fp16
        autocast must not NaN the loss.

        The loss is computed *inside* ``amp_helper.get_autocast_context()``,
        which defaults to fp16 for image-space arms. Pre-fix, ``img * img``
        inside ``compute_ssim_map`` overflowed the fp16 range (max ~6.5e4) for
        any magnitude ≳256 → ``Inf - Inf = NaN`` → the term was NaN-skipped by
        the loss computer, collapsing to a dead loss (dead_loss trap). The fix
        upcasts the SSIM computation to float32 with autocast disabled, so an
        fp16 off-scale input now yields a finite loss and gradient.
        """
        loss_fn = MSSSIMLoss()
        scale = 1e3  # (1e3)^2 = 1e6, far above the fp16 ceiling
        x = (_img() * scale).half().requires_grad_(True)
        y = (_img(seed=5) * scale).half()
        out = loss_fn(x, y)
        assert torch.isfinite(out), "fp16 off-scale MS-SSIM loss went non-finite"
        out.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all(), "fp16 off-scale MS-SSIM grad non-finite"


# ===========================================================================
# Sync-free auto_range (perf regression: 4 .item() GPU syncs per loss call)
# ===========================================================================


def _old_float_resolve(loss_fn: SSIMLoss, img1, img2, override=None) -> float:
    """Pre-fix reference: resolve the data range via .item() Python floats."""
    if override is not None:
        return float(override)
    resolved = float(loss_fn.data_range)
    if not loss_fn.auto_range:
        return max(resolved, 1e-6)
    min_val = min(img1.detach().amin().item(), img2.detach().amin().item())
    max_val = max(img1.detach().amax().item(), img2.detach().amax().item())
    dynamic_range = max_val - min_val
    resolved = (
        max(resolved, dynamic_range) if min_val < 0 else max(resolved, max_val)
    )
    return max(resolved, 1e-6)


class TestSyncFreeDataRange:
    """auto_range resolution stays on-device (0-dim tensors, no .item())."""

    @pytest.mark.parametrize(
        ("scale", "shift"),
        [(1.0, 0.0), (3.0, -1.0), (1e3, 0.0), (0.5, 0.2)],
    )
    def test_equivalence_old_float_vs_new_tensor_path(self, scale, shift):
        """New tensor path reproduces the old .item() float path exactly."""
        torch.manual_seed(0)
        x = torch.rand(2, 1, 32, 32) * scale + shift
        y = torch.rand(2, 1, 32, 32) * scale + shift
        loss_fn = SSIMLoss(auto_range=True)

        new_loss = loss_fn(x, y)
        old_range = _old_float_resolve(loss_fn, x, y)
        old_loss = 1.0 - loss_fn.compute_ssim_with_resolved_range(
            x, y, old_range
        ).mean()

        new_range = loss_fn.resolve_data_range(x, y)
        assert float(new_range) == pytest.approx(old_range)
        assert torch.allclose(new_loss, old_loss)

    def test_auto_range_returns_on_device_scalar_tensor(self):
        loss_fn = SSIMLoss(auto_range=True)
        x = _img()
        y = _img(seed=1)
        resolved = loss_fn.resolve_data_range(x, y)
        assert isinstance(resolved, torch.Tensor)
        assert resolved.ndim == 0
        assert resolved.device == x.device

    def test_explicit_override_still_float(self):
        """Explicit data_range keeps the plain-float fast path."""
        loss_fn = SSIMLoss(auto_range=True)
        x = _img()
        y = _img(seed=1)
        resolved = loss_fn.resolve_data_range(x, y, data_range=2.0)
        assert isinstance(resolved, float) and resolved == 2.0

    def test_no_auto_range_still_float(self):
        loss_fn = SSIMLoss(auto_range=False, data_range=3.0)
        x = _img()
        y = _img(seed=1)
        assert loss_fn.resolve_data_range(x, y) == 3.0

    def test_forward_never_calls_item(self, monkeypatch):
        """Poison Tensor.item: default-configured forward must not sync."""

        def _boom(self):
            raise AssertionError(".item() called in per-step loss path")

        x = _img()
        y = _img(seed=1)
        loss_fn = SSIMLoss(auto_range=True)
        ms_loss_fn = MSSSIMLoss(auto_range=True)
        monkeypatch.setattr(torch.Tensor, "item", _boom)
        out = loss_fn(x, y)
        ms_out = ms_loss_fn(x, y)
        monkeypatch.undo()
        assert torch.isfinite(out)
        assert torch.isfinite(ms_out)

    def test_explicit_data_range_forward_matches_pre_fix(self):
        """Explicit data_range forward numerics are untouched by the rework."""
        x = _img()
        y = _img(seed=1)
        loss_fn = SSIMLoss(auto_range=True)
        out = loss_fn(x, y, data_range=1.0)
        ref = 1.0 - loss_fn.compute_ssim_with_resolved_range(x, y, 1.0).mean()
        assert torch.allclose(out, ref)


# ===========================================================================
# Convenience functions
# ===========================================================================


class TestConvenienceFunctions:

    def test_ssim_loss_function_finite(self):
        x = _img()
        y = _img(seed=5)
        out = ssim_loss(x, y)
        assert torch.isfinite(out)

    def test_msssim_loss_function_finite(self):
        x = _img()
        y = _img(seed=5)
        out = msssim_loss(x, y)
        assert torch.isfinite(out)
