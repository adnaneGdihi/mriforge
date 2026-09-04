"""Unit tests for tropical_semiring_loss.py.

Tropical-semiring consistency loss for the ``tropical_mrf`` /
``tropical_quantitative_maps`` paradigms. The loss genuinely consumes the
tropical-geometry primitive at
:mod:`spectramr.infrastructure.physics.tropical_geometry`, so these tests assert
not only the registry / shape contract but that the primitive is exercised and
that the loss behaves like a *tropical* consistency penalty (zero on identity,
strictly positive on a tropically inconsistent mismatch).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.registry import LossRegistry, create_loss
from spectramr.models.losses.tropical_semiring_loss import (
    TropicalSemiringConsistencyLoss,
)

B, C, H, W = 2, 3, 16, 16


def _img(b=B, c=C, h=H, w=W, seed=0) -> torch.Tensor:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.rand(b, c, h, w, generator=gen)


class TestTropicalSemiringConsistencyLoss:

    # ── Registry ──────────────────────────────────────────────────────────────

    def test_registry_canonical_name(self):
        loss_fn = create_loss("tropical_semiring_consistency")
        assert isinstance(loss_fn, TropicalSemiringConsistencyLoss)

    def test_registry_alias(self):
        loss_fn = create_loss("tropical_mrf_consistency")
        assert isinstance(loss_fn, TropicalSemiringConsistencyLoss)

    def test_registry_domain_is_image(self):
        domains = LossRegistry._loss_domains
        assert domains["tropical_semiring_consistency"]["domain"] == "image"

    # ── Shape / scalar contract ───────────────────────────────────────────────

    def test_returns_scalar_tensor(self):
        loss_fn = TropicalSemiringConsistencyLoss()
        out = loss_fn(_img(seed=1), _img(seed=2))
        assert out.shape == () and torch.isfinite(out)

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_parametric_reduction(self, reduction: str):
        loss_fn = TropicalSemiringConsistencyLoss(reduction=reduction)
        out = loss_fn(_img(seed=1), _img(seed=2))
        assert out.shape == () and torch.isfinite(out)

    # ── Tropical-consistency property ─────────────────────────────────────────

    def test_zero_when_prediction_equals_target(self):
        loss_fn = TropicalSemiringConsistencyLoss()
        x = _img(seed=3)
        out = loss_fn(x, x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_on_mismatch(self):
        """A clearly non-tropical-consistent mismatch yields strictly > 0."""
        loss_fn = TropicalSemiringConsistencyLoss()
        pred = torch.zeros(B, C, H, W)
        target = torch.ones(B, C, H, W) * 5.0
        out = loss_fn(pred, target)
        assert out.item() > 0.0

    def test_nonneg(self):
        loss_fn = TropicalSemiringConsistencyLoss()
        assert loss_fn(_img(seed=4), _img(seed=5)).item() >= 0.0

    def test_gradient_reaches_prediction(self):
        loss_fn = TropicalSemiringConsistencyLoss()
        pred = _img(seed=6).requires_grad_(True)
        target = _img(seed=7)
        loss_fn(pred, target).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()

    # ── Primitive is genuinely exercised ──────────────────────────────────────

    def test_primitive_is_invoked(self, monkeypatch):
        """The loss must actually call log_sum_exp_tropical from the primitive."""
        import spectramr.models.losses.tropical_semiring_loss as mod

        calls = {"n": 0}
        real = mod.log_sum_exp_tropical

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "log_sum_exp_tropical", spy)
        loss_fn = TropicalSemiringConsistencyLoss()
        loss_fn(_img(seed=8), _img(seed=9))
        assert calls["n"] > 0

    def test_active_cone_term_changes_loss(self):
        """The cone/variety term must contribute (toggle via weight)."""
        a = TropicalSemiringConsistencyLoss(cone_weight=0.0)
        b = TropicalSemiringConsistencyLoss(cone_weight=1.0)
        pred, target = _img(seed=10), _img(seed=11)
        assert a(pred, target).item() != b(pred, target).item()

    # ── Raises ────────────────────────────────────────────────────────────────

    def test_raises_shape_mismatch(self):
        loss_fn = TropicalSemiringConsistencyLoss()
        with pytest.raises(ValueError):
            loss_fn(_img(b=2), _img(b=3))

    def test_raises_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction"):
            TropicalSemiringConsistencyLoss(reduction="median")

    def test_raises_negative_cone_weight(self):
        with pytest.raises(ValueError, match="cone_weight"):
            TropicalSemiringConsistencyLoss(cone_weight=-1.0)
