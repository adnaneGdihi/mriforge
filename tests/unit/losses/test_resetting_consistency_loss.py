"""Unit tests for resetting_consistency_loss.py.

The loss genuinely exercises the stochastic-resetting primitive in
``mriforge.models.diffusion.stochastic_resetting``: it penalises the
distance of the coefficient of variation of the prediction-vs-target
residual passage proxy from the criticality value 1 (the Evans-Majumdar
optimal-resetting condition), and the discrepancy between the
``apply_resetting``-perturbed prediction and the target.
"""

from __future__ import annotations

import pytest
import torch

import mriforge.models.diffusion.stochastic_resetting as sr
from mriforge.models.losses.registry import create_loss
from mriforge.models.losses.resetting_consistency_loss import (
    ResettingConsistencyLoss,
)

B, C, H, W = 4, 1, 8, 8


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


class TestResettingConsistencyLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_scalar(self):
        loss_fn = ResettingConsistencyLoss()
        out = loss_fn(_img(seed=1), _img(seed=2))
        assert isinstance(out, torch.Tensor)
        assert out.shape == ()
        assert torch.isfinite(out)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup_canonical(self):
        loss_fn = create_loss("resetting_consistency")
        assert isinstance(loss_fn, ResettingConsistencyLoss)

    @pytest.mark.parametrize(
        "alias", ["stochastic_resetting_consistency", "mfpt_consistency"]
    )
    def test_registry_lookup_aliases(self, alias: str):
        loss_fn = create_loss(alias)
        assert isinstance(loss_fn, ResettingConsistencyLoss)

    # ── Primitive is genuinely exercised ─────────────────────────────────────

    def test_primitive_coefficient_of_variation_called(self, monkeypatch):
        calls = {"n": 0}
        real = sr.coefficient_of_variation

        def spy(samples):
            calls["n"] += 1
            return real(samples)

        monkeypatch.setattr(sr, "coefficient_of_variation", spy)
        loss_fn = ResettingConsistencyLoss()
        loss_fn(_img(seed=1), _img(seed=2))
        assert calls["n"] >= 1

    def test_primitive_apply_resetting_called(self, monkeypatch):
        calls = {"n": 0}
        real = sr.apply_resetting

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sr, "apply_resetting", spy)
        loss_fn = ResettingConsistencyLoss(reset_weight=1.0)
        loss_fn(_img(seed=1), _img(seed=2))
        assert calls["n"] >= 1

    def test_primitive_mfpt_called(self, monkeypatch):
        calls = {"n": 0}
        real = sr.mfpt_with_resetting

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(sr, "mfpt_with_resetting", spy)
        loss_fn = ResettingConsistencyLoss(mfpt_weight=1.0)
        loss_fn(_img(seed=1), _img(seed=2))
        assert calls["n"] >= 1

    # ── Criticality / identity behaviour ─────────────────────────────────────

    def test_criticality_min_when_cov_equals_one(self):
        """The CoV term is minimal (≈0) when the passage proxy has CoV ≈ 1.

        An exponential distribution has CoV exactly 1, so a prediction whose
        residual magnitudes are exponentially distributed should drive the
        criticality term toward zero, below a residual that is constant
        (CoV ≈ 0).
        """
        target = torch.zeros(B, 1, 32, 32)
        gen = torch.Generator()
        gen.manual_seed(7)
        # Exponential residual magnitudes → CoV ≈ 1.
        exp_pred = -torch.log(torch.rand(B, 1, 32, 32, generator=gen).clamp_min(1e-6))
        # Constant residual magnitude → CoV ≈ 0 → |CoV-1| ≈ 1.
        const_pred = torch.full((B, 1, 32, 32), 0.5)

        loss_fn = ResettingConsistencyLoss(
            criticality_weight=1.0, reset_weight=0.0, mfpt_weight=0.0
        )
        loss_exp = loss_fn(exp_pred, target)
        loss_const = loss_fn(const_pred, target)
        assert loss_exp.item() < loss_const.item()

    def test_identity_reset_term_zero_when_pred_equals_target(self):
        """With prediction == target, the reset-consistency discrepancy is 0."""
        x = _img(seed=3)
        loss_fn = ResettingConsistencyLoss(
            criticality_weight=0.0, reset_weight=1.0, mfpt_weight=0.0
        )
        out = loss_fn(x.clone(), x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_away_from_identity(self):
        loss_fn = ResettingConsistencyLoss(
            criticality_weight=0.0, reset_weight=1.0, mfpt_weight=0.0
        )
        out = loss_fn(_img(seed=1), _img(seed=2))
        assert out.item() > 0.0

    # ── Gradient ─────────────────────────────────────────────────────────────

    def test_gradient_reaches_prediction(self):
        pred = _img(seed=1).requires_grad_(True)
        target = _img(seed=2)
        loss_fn = ResettingConsistencyLoss()
        loss_fn(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0.0

    # ── Reduction ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_modes(self, reduction: str):
        loss_fn = ResettingConsistencyLoss(reduction=reduction)
        out = loss_fn(_img(seed=1), _img(seed=2))
        assert torch.isfinite(out)

    # ── Raises ───────────────────────────────────────────────────────────────

    def test_raises_shape_mismatch(self):
        loss_fn = ResettingConsistencyLoss()
        with pytest.raises(ValueError, match="shape"):
            loss_fn(_img(seed=1), torch.rand(B, 1, 16, 16))

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            ResettingConsistencyLoss(reduction="median")

    def test_raises_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            ResettingConsistencyLoss(criticality_weight=-1.0)
