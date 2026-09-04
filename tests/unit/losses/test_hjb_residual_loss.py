"""Unit tests for the Hamilton-Jacobi-Bellman residual loss.

Covers registry membership, scalar-tensor output, the discrepancy
formulation (residual ~0 when prediction satisfies the same HJB relation as
the target, positive otherwise), shape/reduction validation, and -- the
load-bearing assertion -- that the loss genuinely exercises the HJB
primitive ``hjb_optimal_control.hjb_residual_loss`` (the Hamiltonian
operator, not a plain L1/L2 surrogate).
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.hjb_residual_loss import HJBResidualLoss
from spectramr.models.losses.registry import LossRegistry, create_loss


def _value_field(h: int = 16, w: int = 16, t: int = 8) -> torch.Tensor:
    r"""A smooth scalar value-function field V(state_x, state_y, t).

    Returns shape ``[1, T, H, W]`` so the channel axis is the time axis and
    the spatial axes are the (2D) state coordinates.
    """
    ts = torch.linspace(0.0, 1.0, t)
    ys = torch.linspace(0.0, 1.0, h)
    xs = torch.linspace(0.0, 1.0, w)
    gt, gy, gx = torch.meshgrid(ts, ys, xs, indexing="ij")
    # A non-trivial smooth field with both temporal and spatial structure.
    field = 0.5 * (gx**2 + gy**2) + gt
    return field.unsqueeze(0)


class TestRegistration:
    def test_registered_under_canonical_name(self) -> None:
        assert "hjb_residual" in LossRegistry.list_available()

    def test_create_loss_returns_instance(self) -> None:
        loss = create_loss("hjb_residual")
        assert isinstance(loss, HJBResidualLoss)

    def test_domain_metadata_is_image(self) -> None:
        info = LossRegistry._loss_domains["hjb_residual"]
        assert info["domain"] == "image"


class TestOutputContract:
    def test_returns_scalar_tensor(self) -> None:
        loss = HJBResidualLoss()
        field = _value_field()
        out = loss(field, field.clone())
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_zero_when_prediction_equals_target(self) -> None:
        # Discrepancy formulation: identical fields satisfy the same HJB
        # relation, so the residual discrepancy vanishes.
        loss = HJBResidualLoss()
        field = _value_field()
        out = loss(field, field.clone())
        assert torch.isclose(out, torch.zeros(()), atol=1e-5)

    def test_positive_when_prediction_differs(self) -> None:
        loss = HJBResidualLoss()
        target = _value_field()
        pred = target + 0.7 * torch.randn_like(target)
        out = loss(pred, target)
        assert out > 1e-4

    def test_sum_reduction_runs(self) -> None:
        loss = HJBResidualLoss(reduction="sum")
        field = _value_field()
        out = loss(field, field.clone())
        assert out.ndim == 0


class TestValidation:
    def test_shape_mismatch_raises(self) -> None:
        loss = HJBResidualLoss()
        with pytest.raises(ValueError):
            loss(_value_field(16, 16), _value_field(8, 8))

    def test_bad_reduction_raises(self) -> None:
        with pytest.raises(ValueError):
            HJBResidualLoss(reduction="median")

    def test_too_few_dims_raises(self) -> None:
        loss = HJBResidualLoss()
        with pytest.raises(ValueError):
            loss(torch.randn(16, 16), torch.randn(16, 16))


class TestHJBPrimitiveExercised:
    """The HJB Hamiltonian operator must drive the loss -- not plain L1/L2."""

    def test_primitive_is_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import spectramr.models.losses.hjb_residual_loss as mod

        calls = {"n": 0}
        real = mod.hjb_residual_loss

        def _spy(*args: object, **kwargs: object) -> torch.Tensor:
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(mod, "hjb_residual_loss", _spy)
        loss = HJBResidualLoss()
        field = _value_field()
        loss(field + 0.1, field)
        assert calls["n"] >= 1

    def test_gmax_changes_loss(self) -> None:
        # g_max enters only through the min-control Hamiltonian term
        # ``-g_max * ||grad V||``; if the loss ignored the Hamiltonian
        # (e.g. pure L2 of value differences) g_max would have no effect.
        target = _value_field()
        pred = target + 0.3 * torch.randn_like(target)
        lo = HJBResidualLoss(g_max=0.1)(pred, target)
        hi = HJBResidualLoss(g_max=5.0)(pred, target)
        assert not torch.isclose(lo, hi, atol=1e-4)

    def test_constant_field_residual_responds_to_gradient_term(self) -> None:
        # A spatially-constant prediction has zero spatial gradient, so its
        # Hamiltonian min-control term is zero, while a spatially-varying
        # target has a nonzero one -> nonzero discrepancy. This can only be
        # nonzero if the gradient (Hamiltonian) term is actually computed.
        loss = HJBResidualLoss(g_max=2.0)
        target = _value_field()
        const = torch.full_like(target, float(target.mean()))
        out = loss(const, target)
        assert out > 1e-4
