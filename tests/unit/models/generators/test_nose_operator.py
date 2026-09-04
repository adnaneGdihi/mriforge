r"""Unit tests for the Neural-Operator Signal Equation (NOSE).

Targets ``spectramr.models.generators.nose_operator`` and the ``physics_anchor`` loss.

NOSE casts contrast formation as a discretization-invariant neural operator
:math:`\mathcal G_\theta:(\boldsymbol\xi(\cdot),\boldsymbol\varphi)\mapsto
I(\cdot)` over the continuous acquisition manifold, with a physics-anchoring term
:math:`\beta\lVert\mathcal G_\theta(\boldsymbol\xi,\boldsymbol\varphi)-\mathcal
A_{\mathrm{analytic}}(\boldsymbol\xi;\boldsymbol\varphi)\rVert^2`. As
:math:`\beta\to\infty` the minimiser recovers analytic synthetic MRI (the
trusted-physics limit), and by discretization-invariance one trained operator
evaluates at any spatial resolution.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

_ACQ_VEC = torch.tensor([[80.0, 3000.0, 0.0, 90.0, 3.0]])
_ACQ = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}


def _params(h: int, w: int) -> torch.Tensor:
    rho = torch.full((1, 1, h, w), 0.8)
    t1 = torch.full((1, 1, h, w), 900.0)
    t2 = torch.full((1, 1, h, w), 90.0)
    return torch.cat([rho, t1, t2], dim=1)


def test_operator_runs_and_is_discretization_invariant() -> None:
    from spectramr.models.generators.nose_operator import NeuralOperatorSignalEquation

    torch.manual_seed(0)
    op = NeuralOperatorSignalEquation(param_channels=3, acq_dim=5, width=8, modes=4)
    out16 = op(_params(16, 16), _ACQ_VEC)
    out32 = op(_params(32, 32), _ACQ_VEC)  # same operator, different resolution
    assert out16.shape == (1, 1, 16, 16)
    assert out32.shape == (1, 1, 32, 32)


def test_operator_is_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "nose_operator" in MODEL_REGISTRY


def test_physics_anchor_zero_when_prediction_is_analytic_render() -> None:
    from spectramr.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    params = _params(8, 8)
    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    analytic = layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0},
    )
    loss = PhysicsAnchorLoss()(
        prediction=analytic,
        target=torch.zeros_like(analytic),
        context={"params": params, "acquisition": _ACQ, "contrast_type": "spin_echo"},
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_physics_anchor_positive_when_prediction_wrong() -> None:
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    params = _params(8, 8)
    pred = torch.zeros(1, 1, 8, 8)
    loss = PhysicsAnchorLoss()(
        prediction=pred,
        target=torch.zeros_like(pred),
        context={"params": params, "acquisition": _ACQ, "contrast_type": "spin_echo"},
    )
    assert float(loss) > 1e-4


def test_physics_anchor_is_registered() -> None:
    from spectramr.models.losses import create_loss
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    assert isinstance(create_loss("physics_anchor"), PhysicsAnchorLoss)
