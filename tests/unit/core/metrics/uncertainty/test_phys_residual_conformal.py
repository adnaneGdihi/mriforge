"""Unit tests for the PR-CC physics-residual consistency metric.

Targets ``spectramr.core.metrics.uncertainty.phys_residual_conformal``.

The registered metric is the live eval-tier handle: it computes the *mean
physics residual* (a real forward-model consistency scalar, lower is better).
The full calibrated certificate -- coverage and the flagged hallucination map --
is :class:`PhysResidualConformalCertificate` (tested separately), because a
per-batch metric cannot hold a calibration split.
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


def test_metric_is_registered() -> None:
    from spectramr.core.metrics.registry import MetricsRegistry

    assert MetricsRegistry.is_registered("phys_residual_consistency")


def test_zero_for_consistent_params() -> None:
    from spectramr.core.metrics.uncertainty.phys_residual_conformal import (
        PhysResidualConsistencyMetric,
    )

    params = _params()
    target = _render(params)
    metric = PhysResidualConsistencyMetric(
        acquisition=_ACQ, contrast_type="spin_echo"
    )
    assert metric(params, target) == pytest.approx(0.0, abs=1e-5)
    assert metric.higher_is_better is False


def test_positive_for_inconsistent_params() -> None:
    from spectramr.core.metrics.uncertainty.phys_residual_conformal import (
        PhysResidualConsistencyMetric,
    )

    params = _params()
    target = _render(params)
    bad = params.clone()
    bad[:, 2] *= 3.0
    metric = PhysResidualConsistencyMetric(
        acquisition=_ACQ, contrast_type="spin_echo"
    )
    assert metric(bad, target) > 1e-3


def test_acquisition_via_kwargs_overrides_default() -> None:
    from spectramr.core.metrics.uncertainty.phys_residual_conformal import (
        PhysResidualConsistencyMetric,
    )

    params = _params()
    target = _render(params)
    metric = PhysResidualConsistencyMetric()  # no default acquisition
    val = metric(params, target, acquisition=_ACQ, contrast_type="spin_echo")
    assert val == pytest.approx(0.0, abs=1e-5)


def test_missing_acquisition_raises() -> None:
    from spectramr.core.metrics.uncertainty.phys_residual_conformal import (
        PhysResidualConsistencyMetric,
    )

    params = _params()
    target = _render(params)
    metric = PhysResidualConsistencyMetric()  # no acquisition anywhere
    with pytest.raises(ValueError):
        metric(params, target)


def test_import_does_not_pull_infrastructure_calibration() -> None:
    """Layer-direction regression: importing this metric module must NOT import
    ``infrastructure.calibration`` (a leftward import at module scope poisons the
    core.metrics walk-discovery). The ``physics_residual`` call is lazy inside
    ``__call__``; a fresh subprocess proves the import graph stays clean.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import spectramr.core.metrics.uncertainty.phys_residual_conformal\n"
        "assert 'spectramr.infrastructure.calibration.scores' not in sys.modules, (\n"
        "    'module-scope leftward import reintroduced')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
