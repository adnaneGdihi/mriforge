"""Unit tests for sheaf_consistency_loss.py.

The loss must genuinely encode cellular-sheaf consistency by consuming the
primitives in :mod:`mriforge.infrastructure.physics.cellular_sheaf` — it is
*not* a relabelled L1/L2. The key algebraic property under test is that the
sheaf-Laplacian quadratic form is zero iff the section is a global section
(consistent across the cover), and strictly positive otherwise.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.registry import create_loss
from mriforge.models.losses.sheaf_consistency_loss import SheafConsistencyLoss

B, C, H, W = 2, 4, 16, 16


def _rand(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


class TestSheafConsistencyLoss:

    # ── Canary ──────────────────────────────────────────────────────────────

    def test_canary_builds_and_forward_scalar(self):
        loss_fn = SheafConsistencyLoss()
        out = loss_fn(_rand(), _rand(seed=1))
        assert isinstance(out, torch.Tensor)
        assert out.shape == ()
        assert torch.isfinite(out)

    # ── Registry ────────────────────────────────────────────────────────────

    def test_registry_lookup_by_canonical_name(self):
        loss_fn = create_loss("sheaf_consistency")
        assert isinstance(loss_fn, SheafConsistencyLoss)

    def test_registry_domain_is_image(self):
        from mriforge.models.losses.registry import LossRegistry

        info = LossRegistry._loss_domains["sheaf_consistency"]
        assert info["domain"] == "image"

    # ── Property: the sheaf primitive is genuinely exercised ─────────────────

    def test_property_laplacian_zero_for_globally_consistent_section(self):
        """A prediction that is *constant across the cover* (all cover cells
        carry the same section) is a global section ⇒ Laplacian energy = 0,
        so the sheaf term contributes nothing.
        """
        loss_fn = SheafConsistencyLoss(laplacian_weight=1.0, obstruction_weight=0.0)
        # Identical across channels (the cover cells) and constant spatially.
        consistent = torch.ones(B, C, H, W)
        out = loss_fn(consistent, consistent)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_property_laplacian_positive_for_inconsistent_section(self):
        """A prediction whose cover cells disagree has positive sheaf-Laplacian
        energy — strictly larger than the consistent case.
        """
        loss_fn = SheafConsistencyLoss(laplacian_weight=1.0, obstruction_weight=0.0)
        consistent = torch.ones(B, C, H, W)
        inconsistent = _rand()  # channels disagree
        e_consistent = loss_fn(consistent, consistent)
        e_inconsistent = loss_fn(inconsistent, inconsistent)
        assert e_inconsistent.item() > e_consistent.item()
        assert e_inconsistent.item() > 0.0

    def test_property_primitive_is_invoked(self, monkeypatch):
        """Hard proof the loss routes through the sheaf primitive: patch the
        quadratic-form callable and assert it was called.
        """
        import mriforge.models.losses.sheaf_consistency_loss as mod

        calls = {"n": 0}
        real = mod.sheaf_laplacian_quadratic_form

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "sheaf_laplacian_quadratic_form", spy)
        loss_fn = SheafConsistencyLoss(laplacian_weight=1.0)
        loss_fn(_rand(), _rand(seed=2))
        assert calls["n"] > 0

    def test_property_obstruction_invoked(self, monkeypatch):
        import mriforge.models.losses.sheaf_consistency_loss as mod

        calls = {"n": 0}
        real = mod.mayer_vietoris_obstruction_norm

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "mayer_vietoris_obstruction_norm", spy)
        loss_fn = SheafConsistencyLoss(obstruction_weight=1.0)
        loss_fn(_rand(), _rand(seed=3))
        assert calls["n"] > 0

    def test_property_obstruction_zero_when_pred_equals_target(self):
        """Mayer-Vietoris obstruction between identical pred/target is 0."""
        loss_fn = SheafConsistencyLoss(laplacian_weight=0.0, obstruction_weight=1.0)
        x = _rand()
        out = loss_fn(x, x)
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_property_nonneg(self):
        loss_fn = SheafConsistencyLoss()
        assert loss_fn(_rand(), _rand(seed=4)).item() >= 0.0

    def test_property_gradient_reaches_prediction(self):
        pred = _rand().requires_grad_(True)
        loss_fn = SheafConsistencyLoss()
        loss_fn(pred, _rand(seed=5)).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    # ── Reduction ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = SheafConsistencyLoss(reduction=reduction)
        out = loss_fn(_rand(), _rand(seed=6))
        assert torch.isfinite(out)

    def test_property_sum_ge_mean(self):
        x, y = _rand(), _rand(seed=7)
        out_mean = SheafConsistencyLoss(reduction="mean")(x, y)
        out_sum = SheafConsistencyLoss(reduction="sum")(x, y)
        assert out_sum.item() >= out_mean.item()

    # ── Raises ──────────────────────────────────────────────────────────────

    def test_raises_shape_mismatch(self):
        loss_fn = SheafConsistencyLoss()
        with pytest.raises(ValueError, match="shape"):
            loss_fn(_rand(c=4), _rand(c=2))

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            SheafConsistencyLoss(reduction="median")

    def test_raises_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            SheafConsistencyLoss(laplacian_weight=-1.0)
