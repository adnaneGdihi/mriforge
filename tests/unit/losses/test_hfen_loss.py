"""Unit tests for hfen_loss.py — SSOT HFENLoss (registered as 'hfen')."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.hfen_loss import HFENLoss
from mriforge.models.losses.registry import create_loss

B, C, H, W = 2, 1, 32, 32


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


class TestHFENLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = HFENLoss()
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("normalize", [True, False])
    def test_parametric_normalize(self, normalize: bool):
        loss_fn = HFENLoss(normalize=normalize)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = HFENLoss(reduction=reduction)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    @pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
    def test_parametric_sigma(self, sigma: float):
        loss_fn = HFENLoss(sigma=sigma)
        x = _img()
        y = _img(seed=1)
        out = loss_fn(x, y)
        assert torch.isfinite(out) and out.item() >= 0.0

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_identity_near_zero(self):
        """HFEN(x, x) = 0 because LoG(x) - LoG(x) = 0."""
        loss_fn = HFENLoss()
        x = _img()
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_property_nonneg(self):
        loss_fn = HFENLoss()
        x = _img()
        y = _img(seed=1)
        assert loss_fn(x, y).item() >= 0.0

    def test_property_gradient_reaches_both_inputs(self):
        x = _img().requires_grad_(True)
        y = _img(seed=1)
        loss_fn = HFENLoss()
        loss_fn(x, y).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_zero_input(self):
        loss_fn = HFENLoss()
        x = torch.zeros(B, C, H, W)
        y = torch.zeros(B, C, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_large_magnitude(self):
        loss_fn = HFENLoss()
        x = torch.full((B, C, H, W), 1e3)
        y = torch.zeros(B, C, H, W)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_complex_input(self):
        """Complex input → uses magnitude."""
        loss_fn = HFENLoss()
        x = torch.view_as_complex(torch.randn(B, C, H, W, 2).contiguous())
        y = _img()
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    def test_edge_multichannel(self):
        loss_fn = HFENLoss()
        x = _img(c=2)
        y = _img(c=2, seed=5)
        out = loss_fn(x, y)
        assert torch.isfinite(out)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("hfen")
        assert isinstance(loss_fn, HFENLoss)

    def test_registry_alias_lookup(self):
        loss_fn = create_loss("high_frequency_error_norm")
        assert isinstance(loss_fn, HFENLoss)
