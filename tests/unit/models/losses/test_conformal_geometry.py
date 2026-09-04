"""Unit tests for ConformalModulusLoss.

Since 2026-07 the underlying ``conformal_modulus_quadrilateral`` computes a
**genuine** conformal modulus (Dirichlet energy of the harmonic field solved on
the physical quad by FE — shape-dependent; see
``tests/unit/physics/test_conformal_geometry.py::
test_conformal_modulus_matches_rectangle_aspect_ratio``), scaled by each image's
graph metric ``w = sqrt(1 + |∇I|²)`` so the penalty is also non-degenerate
across reconstructions (the 2026-06 fix; before it both terms were computed on a
fixed canonical square and the residual was identically zero). These tests pin
positivity, image-dependence, and registration of the loss.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.losses.conformal_geometry import ConformalModulusLoss  # noqa: E402
from spectramr.models.losses.registry import list_available  # noqa: E402


def _loss() -> ConformalModulusLoss:
    return ConformalModulusLoss(weight=1.0, grid_size=16)


def test_registered() -> None:
    assert "conformal_modulus" in list_available()


def test_nonzero_for_different_images() -> None:
    torch.manual_seed(0)
    loss = _loss()
    pred = torch.randn(2, 1, 32, 32)
    target = torch.randn(2, 1, 32, 32)
    out = loss(pred, target)["loss"]
    assert out.item() > 1e-6


def test_zero_for_identical_images() -> None:
    loss = _loss()
    x = torch.randn(2, 1, 32, 32)
    out = loss(x, x)["loss"]
    assert out.item() == pytest.approx(0.0, abs=1e-6)


def test_default_quads_used_when_none_supplied() -> None:
    """No ``quad_corners`` in context -> built-in multi-scale templates fire,
    so the loss is non-degenerate without any caller wiring (the recon
    strategies never inject quads)."""
    torch.manual_seed(1)
    loss = _loss()
    out = loss(torch.randn(1, 1, 32, 32), torch.randn(1, 1, 32, 32))
    assert out["loss"].item() > 1e-6
    assert out["conformal_modulus"].item() > 1e-6


def test_supplied_quad_templates_are_used() -> None:
    """A caller may override with explicit ``[K, 4, 2]`` quad templates."""
    torch.manual_seed(2)
    loss = _loss()
    quads = torch.tensor([[[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]])
    out = loss(
        torch.randn(2, 1, 32, 32),
        torch.randn(2, 1, 32, 32),
        quad_corners=quads,
    )["loss"]
    assert out.item() > 1e-6


def test_gradients_flow_to_prediction() -> None:
    loss = _loss()
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    loss(pred, target)["loss"].backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # Stronger than "grad exists": a zero grad would pass `is not None` yet mean
    # the loss is inert. Pin that the gradient actually carries signal.
    assert torch.any(pred.grad != 0)


# ---------------------------------------------------------------------------
# TeichmullerGeodesicRegularisation — fail loud on a missing schedule
# ---------------------------------------------------------------------------


def test_teichmuller_raises_without_r_schedule() -> None:
    """The registered Teichmüller loss must RAISE when ``r_schedule`` is absent
    or too short, not silently return a zero dict (an inert loss no-op is a
    facade — pitfall #16)."""
    from spectramr.models.losses.conformal_geometry import (
        TeichmullerGeodesicRegularisation,
    )

    loss = TeichmullerGeodesicRegularisation(weight=1.0)
    pred = torch.randn(2, 1, 8, 8)
    with pytest.raises(ValueError, match="r_schedule"):
        loss(pred, pred)  # no r_schedule in context
    with pytest.raises(ValueError, match="r_schedule"):
        loss(pred, pred, r_schedule=torch.tensor([0.1, 0.2]))  # numel < 3


def test_teichmuller_computes_with_valid_schedule() -> None:
    from spectramr.models.losses.conformal_geometry import (
        TeichmullerGeodesicRegularisation,
    )

    loss = TeichmullerGeodesicRegularisation(weight=1.0)
    pred = torch.randn(2, 1, 8, 8)
    r = torch.linspace(0.1, 0.8, 8)
    out = loss(pred, pred, r_schedule=r)
    assert out["loss"].item() >= 0.0
    assert "teichmuller_geodesic" in out
