"""Unit tests for the Helmholtz-Hodge decomposition motion-field loss.

Covers registry membership, scalar-tensor output, zero loss on identity,
shape/channel validation, and -- crucially -- that the loss exercises the
Hodge decomposition: a purely solenoidal (divergence-free) field carries
~0 divergence penalty while an irrotational (gradient) field does not.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.losses.hodge_decomposition_loss import (
    HodgeDecompositionLoss,
)
from spectramr.models.losses.registry import LossRegistry, create_loss


def _grid(h: int = 16, w: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Return meshgrid coordinates in [0, 1)."""
    ys = torch.linspace(0.0, 1.0, h)
    xs = torch.linspace(0.0, 1.0, w)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return gy, gx


def _gradient_field(h: int = 16, w: int = 16) -> torch.Tensor:
    r"""Irrotational field v = grad(phi), phi = (x^2 + y^2) / 2.

    Then u = dphi/dy = y, v = dphi/dx = x; curl ~ 0, divergence ~ 2.
    Returns shape ``[1, 2, H, W]``.
    """
    gy, gx = _grid(h, w)
    u = gy  # vy component
    v = gx  # vx component
    return torch.stack([u, v], dim=0).unsqueeze(0)


def _solenoidal_field(h: int = 16, w: int = 16) -> torch.Tensor:
    r"""Divergence-free rotational field: u = -(x - 0.5), v = (y - 0.5).

    div = d(-(x-.5))/dx + d((y-.5))/dy ... but with axis convention
    (vy=u indexed along H, vx=v indexed along W) we want
    div = dvy/dy + dvx/dx = 0. Use u = -(x-.5) [varies along W, not H]
    and v = (y-.5) [varies along H, not W] so both partials vanish.
    Returns shape ``[1, 2, H, W]``.
    """
    gy, gx = _grid(h, w)
    u = -(gx - 0.5)  # vy, varies along W only -> dvy/dy = 0
    v = gy - 0.5  # vx, varies along H only -> dvx/dx = 0
    return torch.stack([u, v], dim=0).unsqueeze(0)


class TestRegistration:
    def test_registered_under_canonical_name(self) -> None:
        assert "hodge_decomposition" in LossRegistry.list_available()

    def test_create_loss_returns_instance(self) -> None:
        loss = create_loss("hodge_decomposition")
        assert isinstance(loss, HodgeDecompositionLoss)

    def test_domain_metadata_is_image(self) -> None:
        info = LossRegistry._loss_domains["hodge_decomposition"]
        assert info["domain"] == "image"


class TestOutputContract:
    def test_returns_scalar_tensor(self) -> None:
        loss = HodgeDecompositionLoss()
        field = torch.randn(2, 2, 16, 16)
        out = loss(field, field.clone())
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 0

    def test_zero_when_prediction_equals_target(self) -> None:
        loss = HodgeDecompositionLoss()
        field = torch.randn(2, 2, 16, 16)
        out = loss(field, field.clone())
        assert torch.isclose(out, torch.zeros(()), atol=1e-5)

    def test_sum_reduction_runs(self) -> None:
        loss = HodgeDecompositionLoss(reduction="sum")
        field = torch.randn(1, 2, 16, 16)
        out = loss(field, field.clone())
        assert out.ndim == 0


class TestValidation:
    def test_shape_mismatch_raises(self) -> None:
        loss = HodgeDecompositionLoss()
        with pytest.raises(ValueError):
            loss(torch.randn(1, 2, 16, 16), torch.randn(1, 2, 8, 8))

    def test_single_channel_raises(self) -> None:
        loss = HodgeDecompositionLoss()
        with pytest.raises(ValueError):
            loss(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))

    def test_bad_reduction_raises(self) -> None:
        with pytest.raises(ValueError):
            HodgeDecompositionLoss(reduction="median")


class TestHodgeStructure:
    """The decomposition must actually drive the penalty -- not L1/L2."""

    def test_divergence_penalty_distinguishes_field_types(self) -> None:
        # Incompressibility (divergence) penalty only: solenoidal field
        # should score ~0; gradient field should score clearly nonzero.
        loss = HodgeDecompositionLoss(
            divergence_weight=1.0,
            component_match_weight=0.0,
        )
        zero = torch.zeros(1, 2, 16, 16)

        sol = _solenoidal_field()
        grad = _gradient_field()

        # Penalise spurious divergence in the *prediction* relative to a
        # divergence-free target (zero field has zero divergence).
        pen_sol = loss(sol, zero)
        pen_grad = loss(grad, zero)

        assert pen_sol < 1e-3, f"solenoidal field divergence penalty too high: {pen_sol}"
        assert pen_grad > 10.0 * pen_sol + 1e-3, (
            f"gradient field penalty ({pen_grad}) should dominate "
            f"solenoidal ({pen_sol})"
        )

    def test_component_match_uses_decomposition(self) -> None:
        # With component-match weight on, matching a field to itself is 0,
        # but matching a solenoidal field to a gradient field is nonzero
        # because their decomposed parts differ.
        loss = HodgeDecompositionLoss(
            divergence_weight=0.0,
            component_match_weight=1.0,
        )
        sol = _solenoidal_field()
        grad = _gradient_field()
        same = loss(sol, sol.clone())
        diff = loss(sol, grad)
        assert torch.isclose(same, torch.zeros(()), atol=1e-5)
        assert diff > 1e-3
