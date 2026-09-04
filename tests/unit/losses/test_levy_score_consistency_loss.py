"""Unit tests for LevyScoreConsistencyLoss (alpha-stable Levy score matching)."""

from __future__ import annotations

import torch

from spectramr.models.losses.levy_score_consistency_loss import (
    LevyScoreConsistencyLoss,
)


class TestRegistration:
    """The loss must be discoverable through the loss registry."""

    def test_registered_under_canonical_name(self):
        from spectramr.models.losses import create_loss, list_available

        assert "levy_score_consistency" in list_available()
        loss = create_loss("levy_score_consistency")
        assert isinstance(loss, LevyScoreConsistencyLoss)

    def test_registered_domain_is_image(self):
        from spectramr.models.losses.registry import LossRegistry

        meta = LossRegistry._loss_domains["levy_score_consistency"]
        assert meta["domain"] == "image"


class TestForward:
    """Forward-pass contract."""

    def test_returns_scalar_tensor(self):
        loss_fn = LevyScoreConsistencyLoss(alpha=1.5)
        pred = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)
        val = loss_fn(pred, target)
        assert isinstance(val, torch.Tensor)
        assert val.shape == ()
        assert torch.isfinite(val)

    def test_shape_mismatch_raises_value_error(self):
        loss_fn = LevyScoreConsistencyLoss(alpha=1.5)
        pred = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 8, 8)
        try:
            loss_fn(pred, target)
        except ValueError:
            return
        raise AssertionError("expected ValueError on shape mismatch")

    def test_invalid_alpha_raises(self):
        for bad in (0.0, -1.0, 2.5):
            try:
                LevyScoreConsistencyLoss(alpha=bad)
            except ValueError:
                continue
            raise AssertionError(f"alpha={bad} should raise")


class TestPrimitiveExercised:
    """The loss MUST genuinely route through the alpha-stable primitive."""

    def test_fractional_score_matching_loss_is_called(self, monkeypatch):
        import spectramr.models.losses.levy_score_consistency_loss as mod

        calls = {}
        real = mod.fractional_score_matching_loss

        def spy(score_model, x0, **kwargs):
            calls["alpha"] = kwargs.get("alpha")
            calls["count"] = calls.get("count", 0) + 1
            return real(score_model, x0, **kwargs)

        monkeypatch.setattr(mod, "fractional_score_matching_loss", spy)

        loss_fn = LevyScoreConsistencyLoss(alpha=1.7, sigma=0.5)
        pred = torch.rand(1, 1, 16, 16)
        target = torch.rand(1, 1, 16, 16)
        loss_fn(pred, target)

        assert calls.get("count", 0) >= 1
        assert calls["alpha"] == 1.7


class TestBehavior:
    """Gradient flow and alpha-sensitivity."""

    def test_gradient_flows_to_prediction(self):
        loss_fn = LevyScoreConsistencyLoss(alpha=1.5)
        pred = torch.rand(1, 1, 16, 16, requires_grad=True)
        target = torch.rand(1, 1, 16, 16)
        val = loss_fn(pred, target)
        val.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0

    def test_alpha_two_differs_from_alpha_one_point_five(self):
        pred = torch.rand(2, 1, 16, 16)
        target = torch.rand(2, 1, 16, 16)

        gen_a = torch.Generator().manual_seed(123)
        gen_b = torch.Generator().manual_seed(123)
        gauss = LevyScoreConsistencyLoss(alpha=2.0, generator=gen_a)
        heavy = LevyScoreConsistencyLoss(alpha=1.5, generator=gen_b)

        v_gauss = gauss(pred, target)
        v_heavy = heavy(pred, target)

        assert torch.isfinite(v_gauss)
        assert torch.isfinite(v_heavy)
        assert not torch.allclose(v_gauss, v_heavy)
