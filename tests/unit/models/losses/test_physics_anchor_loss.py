"""Unit tests for the physics-anchoring (NOSE) loss.

Targets ``spectramr.models.losses.physics_anchor_loss.PhysicsAnchorLoss``.

Covers the loss contract (registered; anchors an operator output to the analytic
Bloch render), the required-context guard, and the layer-direction regression:
the module must not import ``infrastructure.calibration`` at module scope.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit

_ACQ = {"TR": 3000.0, "TE": 90.0, "TI": 0.0}


def _params() -> torch.Tensor:
    rho = torch.tensor([[[[1.0, 0.8], [0.6, 0.9]]]])
    t1 = torch.tensor([[[[800.0, 1200.0], [600.0, 1000.0]]]])
    t2 = torch.tensor([[[[80.0, 120.0], [60.0, 100.0]]]])
    return torch.cat([rho, t1, t2], dim=1)


def _render(params: torch.Tensor) -> torch.Tensor:
    from spectramr.infrastructure.physics.multi_physics_bloch import (
        MultiPhysicsBlochLayer,
    )

    layer = MultiPhysicsBlochLayer(learnable_flip_angle=False)
    return layer(
        {"rho": params[:, 0:1], "t1": params[:, 1:2], "t2": params[:, 2:3]},
        {"TR": 3000.0, "TE": 90.0, "TI": 0.0, "contrast_type": 0},
    )


def test_loss_is_registered() -> None:
    from spectramr.models.losses.registry import LossRegistry

    assert "physics_anchor" in LossRegistry.list_available()


def test_zero_when_prediction_matches_analytic_render() -> None:
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    params = _params()
    prediction = _render(params)
    loss = PhysicsAnchorLoss(weight=1.0)
    value = loss(
        prediction,
        torch.zeros_like(prediction),
        context={"params": params, "acquisition": _ACQ, "contrast_type": "spin_echo"},
    )
    assert value.item() == pytest.approx(0.0, abs=1e-8)


def test_positive_when_prediction_deviates() -> None:
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    params = _params()
    prediction = _render(params) + 0.5
    loss = PhysicsAnchorLoss(weight=2.0)
    value = loss(
        prediction,
        torch.zeros_like(prediction),
        context={"params": params, "acquisition": _ACQ},
    )
    assert value.item() > 0.0


def test_missing_context_raises() -> None:
    from spectramr.models.losses.physics_anchor_loss import PhysicsAnchorLoss

    loss = PhysicsAnchorLoss()
    x = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError):
        loss(x, x, context=None)
    with pytest.raises(ValueError):
        loss(x, x, context={"params": _params()})  # no acquisition


def test_import_does_not_pull_infrastructure_calibration() -> None:
    """Layer-direction regression: models/ must not import infrastructure/ at
    module scope (calibration.scores is not physics-SSOT-exempt)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import spectramr.models.losses.physics_anchor_loss\n"
        "assert 'spectramr.infrastructure.calibration.scores' not in sys.modules, (\n"
        "    'module-scope leftward import reintroduced')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
