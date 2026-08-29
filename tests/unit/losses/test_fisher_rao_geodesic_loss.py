"""Unit tests for the Fisher-Rao geodesic information-geometry loss.

Covers registry membership, scalar-tensor output, near-zero loss when
prediction == target, strict positivity otherwise, and -- crucially -- that
the loss genuinely encodes the Fisher-Rao metric rather than a plain
Euclidean residual: it must exercise the
``infrastructure.physics.fisher_rao_geometry`` primitive and respond to the
Fisher-Rao scaling (a variance-distance is down-weighted by the local
metric, so the loss differs from plain L2 on the same displacement).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.models.losses.fisher_rao_geodesic_loss import (
    FisherRaoGeodesicLoss,
)
from mriforge.models.losses.registry import LossRegistry, create_loss


def _img(mean: float, std: float, h: int = 16, w: int = 16) -> torch.Tensor:
    """Deterministic [1, 1, H, W] field with a controlled mean/std."""
    g = torch.linspace(-1.0, 1.0, h * w).reshape(1, 1, h, w)
    g = (g - g.mean()) / (g.std() + 1e-8)
    return g * std + mean


class TestRegistration:
    def test_registered_under_canonical_name(self) -> None:
        assert "fisher_rao_geodesic" in LossRegistry.list_available()

    def test_create_loss_returns_instance(self) -> None:
        loss = create_loss("fisher_rao_geodesic")
        assert isinstance(loss, FisherRaoGeodesicLoss)

    def test_aliases_resolve(self) -> None:
        for alias in ("fisher_rao_flow", "fr_geodesic", "information_geometry_geodesic"):
            assert isinstance(create_loss(alias), FisherRaoGeodesicLoss)

    def test_domain_is_image(self) -> None:
        meta = LossRegistry._loss_domains["fisher_rao_geodesic"]
        assert meta["domain"] == "image"


class TestForwardContract:
    def test_returns_scalar_tensor(self) -> None:
        loss = FisherRaoGeodesicLoss()
        out = loss(_img(0.5, 0.3), _img(0.7, 0.4))
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_zero_when_identical(self) -> None:
        loss = FisherRaoGeodesicLoss()
        x = _img(0.5, 0.3)
        out = loss(x, x.clone())
        assert out.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_different(self) -> None:
        loss = FisherRaoGeodesicLoss()
        out = loss(_img(0.2, 0.3), _img(0.9, 0.6))
        assert out.item() > 0.0

    def test_shape_mismatch_raises(self) -> None:
        loss = FisherRaoGeodesicLoss()
        with pytest.raises(ValueError):
            loss(_img(0.5, 0.3, h=16, w=16), _img(0.5, 0.3, h=8, w=8))

    def test_invalid_reduction_raises(self) -> None:
        with pytest.raises(ValueError):
            FisherRaoGeodesicLoss(reduction="median")

    def test_sum_reduction_differs_from_mean(self) -> None:
        a = torch.cat([_img(0.1, 0.2), _img(0.8, 0.5)], dim=0)
        b = torch.cat([_img(0.4, 0.3), _img(0.2, 0.7)], dim=0)
        mean_l = FisherRaoGeodesicLoss(reduction="mean")(a, b)
        sum_l = FisherRaoGeodesicLoss(reduction="sum")(a, b)
        # sum over batch of 2 strictly positive terms exceeds their mean.
        assert sum_l.item() > mean_l.item()

    def test_is_differentiable(self) -> None:
        loss = FisherRaoGeodesicLoss()
        pred = _img(0.3, 0.4).requires_grad_(True)
        out = loss(pred, _img(0.7, 0.2))
        out.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


class TestFisherRaoGeometry:
    """The loss must honour the Fisher-Rao metric, not Euclidean distance."""

    def test_primitive_is_exercised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fisher_norm from the primitive must be called during forward."""
        import mriforge.models.losses.fisher_rao_geodesic_loss as mod

        calls = {"n": 0}
        real = mod.fisher_norm

        def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "fisher_norm", spy)
        FisherRaoGeodesicLoss()(_img(0.2, 0.3), _img(0.6, 0.5))
        assert calls["n"] > 0

    def test_metric_uses_fisher_information_primitive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import mriforge.models.losses.fisher_rao_geodesic_loss as mod

        calls = {"n": 0}
        real = mod.fisher_information_from_jacobian

        def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "fisher_information_from_jacobian", spy)
        FisherRaoGeodesicLoss()(_img(0.2, 0.3), _img(0.6, 0.5))
        assert calls["n"] > 0

    def test_metric_consistent_not_euclidean(self) -> None:
        r"""Fisher-Rao curves the parameter space: a fixed Euclidean
        displacement in (mean, log-std) costs *less* when the local
        std (the natural scale) is large. So two pairs with the SAME
        Euclidean (mean, std) gap but different std baselines yield
        DIFFERENT Fisher-Rao distances -- a plain L2 cannot do this.
        """
        loss = FisherRaoGeodesicLoss()
        # Pair A: tight distributions (small std baseline).
        a_pred, a_tgt = _img(0.0, 0.10), _img(0.20, 0.20)
        # Pair B: same mean/std *gaps* (0.20, 0.10) but wider baseline.
        b_pred, b_tgt = _img(0.0, 0.80), _img(0.20, 0.90)
        d_a = loss(a_pred, a_tgt).item()
        d_b = loss(b_pred, b_tgt).item()
        assert d_a != pytest.approx(d_b, rel=1e-3)
        # The Fisher metric on the mean coordinate scales as 1/sigma^2, so
        # the tighter pair (small sigma) must cost MORE for the same gap.
        assert d_a > d_b

    def test_scales_with_distribution_gap(self) -> None:
        loss = FisherRaoGeodesicLoss()
        near = loss(_img(0.5, 0.3), _img(0.55, 0.32)).item()
        far = loss(_img(0.5, 0.3), _img(0.95, 0.70)).item()
        assert far > near
