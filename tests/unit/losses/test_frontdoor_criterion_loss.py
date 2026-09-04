"""Unit tests for the Pearl front-door criterion reconstruction loss.

Covers registry membership, scalar-tensor output, near-zero loss when the
prediction equals the target (so the front-door-adjusted estimate matches),
strict positivity otherwise, the mean/sum reduction contract, shape-mismatch
guarding, and -- crucially -- that the loss genuinely exercises the
``infrastructure.physics.causal_frontdoor`` primitive rather than collapsing
to a plain L1/L2 residual. The front-door adjustment must move the loss away
from the Euclidean residual whenever the mediator carries scanner/site
information (non-trivial MI(M;S)).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.frontdoor_criterion_loss import (
    FrontdoorCriterionLoss,
)
from spectramr.models.losses.registry import LossRegistry, create_loss


def _img(mean: float, std: float, h: int = 16, w: int = 16) -> torch.Tensor:
    """Deterministic [1, 1, H, W] field with a controlled mean/std."""
    g = torch.linspace(-1.0, 1.0, h * w).reshape(1, 1, h, w)
    g = (g - g.mean()) / (g.std() + 1e-8)
    return g * std + mean


class TestRegistration:
    def test_registered_under_canonical_name(self) -> None:
        assert "frontdoor_criterion" in LossRegistry.list_available()

    def test_create_loss_returns_instance(self) -> None:
        loss = create_loss("frontdoor_criterion")
        assert isinstance(loss, FrontdoorCriterionLoss)

    def test_aliases_resolve(self) -> None:
        for alias in ("front_door_adjustment", "pearl_frontdoor", "frontdoor_federated_loss"):
            assert isinstance(create_loss(alias), FrontdoorCriterionLoss)

    def test_domain_is_image(self) -> None:
        meta = LossRegistry._loss_domains["frontdoor_criterion"]
        assert meta["domain"] == "image"


class TestForwardContract:
    def test_returns_scalar_tensor(self) -> None:
        loss = FrontdoorCriterionLoss()
        out = loss(_img(0.5, 0.3), _img(0.7, 0.4))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_zero_when_identical(self) -> None:
        loss = FrontdoorCriterionLoss()
        x = _img(0.5, 0.3)
        out = loss(x, x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_different(self) -> None:
        loss = FrontdoorCriterionLoss()
        out = loss(_img(0.2, 0.3), _img(0.9, 0.6))
        assert out.item() > 0.0

    def test_shape_mismatch_raises(self) -> None:
        loss = FrontdoorCriterionLoss()
        with pytest.raises(ValueError):
            loss(_img(0.5, 0.3, h=16, w=16), _img(0.5, 0.3, h=8, w=8))

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError):
            FrontdoorCriterionLoss(reduction="median")

    def test_sum_reduction_differs_from_mean(self) -> None:
        a = torch.cat([_img(0.1, 0.2), _img(0.8, 0.5)], dim=0)
        b = torch.cat([_img(0.4, 0.3), _img(0.2, 0.7)], dim=0)
        mean_l = FrontdoorCriterionLoss(reduction="mean")(a, b)
        sum_l = FrontdoorCriterionLoss(reduction="sum")(a, b)
        assert sum_l.item() > mean_l.item()

    def test_is_differentiable(self) -> None:
        loss = FrontdoorCriterionLoss()
        pred = _img(0.3, 0.4).requires_grad_(True)
        out = loss(pred, _img(0.7, 0.2))
        out.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestFrontdoorCriterion:
    """The loss must honour the Pearl front-door primitive, not plain L2."""

    def test_adjusted_prediction_primitive_is_exercised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """frontdoor_adjusted_prediction from the primitive must be called."""
        import spectramr.models.losses.frontdoor_criterion_loss as mod

        calls = {"n": 0}
        real = mod.frontdoor_adjusted_prediction

        def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "frontdoor_adjusted_prediction", spy)
        FrontdoorCriterionLoss()(_img(0.2, 0.3), _img(0.6, 0.5))
        assert calls["n"] > 0

    def test_mi_primitive_is_exercised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """donsker_varadhan_mi must be called to penalise mediator-scanner MI."""
        import spectramr.models.losses.frontdoor_criterion_loss as mod

        calls = {"n": 0}
        real = mod.donsker_varadhan_mi

        def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "donsker_varadhan_mi", spy)
        FrontdoorCriterionLoss()(_img(0.2, 0.3), _img(0.6, 0.5))
        assert calls["n"] > 0

    def test_scanner_kwarg_changes_loss(self) -> None:
        """A scanner/site label that correlates with the mediator must move
        the loss away from the scanner-agnostic baseline -- proving the
        front-door (confounder-robust) adjustment is actually applied."""
        loss = FrontdoorCriterionLoss()
        pred = torch.stack(
            [_img(0.1, 0.3)[0], _img(0.9, 0.4)[0]], dim=0
        )
        target = torch.stack(
            [_img(0.2, 0.3)[0], _img(0.8, 0.5)[0]], dim=0
        )
        scanner_a = torch.tensor([0, 0])
        scanner_b = torch.tensor([0, 1])
        out_a = loss(pred, target, scanner=scanner_a).item()
        out_b = loss(pred, target, scanner=scanner_b).item()
        assert out_a != pytest.approx(out_b, rel=1e-4)

    def test_not_equal_to_plain_l2(self) -> None:
        """With a non-degenerate mediator/scanner split the front-door loss
        must differ from a bare MSE on the same (pred, target). The causal
        adjustment is only meaningful across a batch of observations (a single
        sample yields a degenerate one-bin conditional), so use a real batch."""
        loss = FrontdoorCriterionLoss(mi_weight=1.0)
        pred = torch.cat([_img(0.2, 0.3), _img(0.7, 0.5), _img(0.4, 0.2)], dim=0)
        target = torch.cat([_img(0.7, 0.6), _img(0.3, 0.4), _img(0.9, 0.3)], dim=0)
        plain_mse = (pred - target).pow(2).mean().item()
        out = loss(pred, target).item()
        assert out != pytest.approx(plain_mse, rel=1e-4)

    def test_scales_with_residual_gap(self) -> None:
        loss = FrontdoorCriterionLoss()
        near = loss(_img(0.5, 0.3), _img(0.55, 0.32)).item()
        far = loss(_img(0.5, 0.3), _img(0.95, 0.70)).item()
        assert far > near
