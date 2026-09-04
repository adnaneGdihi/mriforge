"""Unit tests for bloch_signal_synthesis_consistency_loss.py."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.bloch_signal_synthesis_consistency_loss import (
    BlochSignalSynthesisConsistencyLoss,
)
from spectramr.models.losses.registry import create_loss

B, H, W = 2, 16, 16

_PARAMS_2 = [
    {"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 15.0},
    {"TR_ms": 1200.0, "TE_ms": 24.0, "FA_deg": 30.0},
]


def _maps(b=B, h=H, w=W, seed=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    t1 = torch.rand(b, 1, h, w, generator=gen) * 1000.0 + 200.0  # [200, 1200] ms
    t2 = torch.rand(b, 1, h, w, generator=gen) * 100.0 + 10.0    # [10, 110] ms
    pd = torch.rand(b, 1, h, w, generator=gen)
    return t1, t2, pd


class TestBlochSignalSynthesisConsistencyLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_finite(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1, t2, pd = _maps()
        observed = torch.rand(B, 2, H, W)  # 2 contrasts
        out = loss_fn(t1, t2, pd, observed, _PARAMS_2)
        assert torch.isfinite(out) and out.shape == ()

    # ── Parametric ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("weight", [0.5, 1.0, 2.0])
    def test_parametric_weight(self, weight: float):
        loss_fn = BlochSignalSynthesisConsistencyLoss(weight=weight)
        t1, t2, pd = _maps()
        observed = torch.rand(B, 2, H, W)
        out = loss_fn(t1, t2, pd, observed, _PARAMS_2)
        assert torch.isfinite(out) and out.item() >= 0.0

    # ── Property ────────────────────────────────────────────────────────────

    def test_property_nonneg(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1, t2, pd = _maps()
        observed = torch.rand(B, 2, H, W)
        assert loss_fn(t1, t2, pd, observed, _PARAMS_2).item() >= 0.0

    def test_property_zero_weight_returns_zero(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss(weight=0.0)
        t1, t2, pd = _maps()
        observed = torch.rand(B, 2, H, W)
        out = loss_fn(t1, t2, pd, observed, _PARAMS_2)
        assert out.item() == pytest.approx(0.0, abs=1e-9)

    def test_property_gradient_reaches_t1(self):
        t1, t2, pd = _maps()
        t1 = t1.requires_grad_(True)
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        observed = torch.rand(B, 2, H, W)
        loss_fn(t1, t2, pd, observed, _PARAMS_2).backward()
        assert t1.grad is not None and torch.isfinite(t1.grad).all()

    # ── Edge ────────────────────────────────────────────────────────────────

    def test_edge_single_contrast(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1, t2, pd = _maps()
        observed = torch.rand(B, 1, H, W)
        params = [{"TR_ms": 600.0, "TE_ms": 12.0, "FA_deg": 15.0}]
        out = loss_fn(t1, t2, pd, observed, params)
        assert torch.isfinite(out)

    def test_edge_large_t_values(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1 = torch.full((B, 1, H, W), 5000.0)
        t2 = torch.full((B, 1, H, W), 2000.0)
        pd = torch.ones(B, 1, H, W)
        observed = torch.rand(B, 2, H, W)
        out = loss_fn(t1, t2, pd, observed, _PARAMS_2)
        assert torch.isfinite(out)

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_param_count_mismatch(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1, t2, pd = _maps()
        observed = torch.rand(B, 2, H, W)
        # Only 1 param dict for 2 contrast channels
        with pytest.raises(ValueError, match="acquisition_params"):
            loss_fn(t1, t2, pd, observed, [_PARAMS_2[0]])

    def test_raises_missing_param_key(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        t1, t2, pd = _maps()
        observed = torch.rand(B, 1, H, W)
        bad_params = [{"TR_ms": 600.0}]  # missing TE_ms, FA_deg
        with pytest.raises(ValueError):
            loss_fn(t1, t2, pd, observed, bad_params)

    def test_raises_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            BlochSignalSynthesisConsistencyLoss(weight=-1.0)

    def test_raises_wrong_t1_shape(self):
        loss_fn = BlochSignalSynthesisConsistencyLoss()
        bad_t1 = torch.rand(B, 2, H, W)  # Should be [B, 1, H, W]
        t2, pd = torch.rand(B, 1, H, W), torch.rand(B, 1, H, W)
        observed = torch.rand(B, 2, H, W)
        with pytest.raises(ValueError):
            loss_fn(bad_t1, t2, pd, observed, _PARAMS_2)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup(self):
        loss_fn = create_loss("bloch_signal_synthesis_consistency")
        assert isinstance(loss_fn, BlochSignalSynthesisConsistencyLoss)
