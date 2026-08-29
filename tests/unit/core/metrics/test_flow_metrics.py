"""Unit tests for core/metrics/flow_metrics.py.

Covers: ConsistencyOfMass, Divergence.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.core.metrics.flow_metrics import ConsistencyOfMass, Divergence


# ---------------------------------------------------------------------------
# Canary
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mass_conservation_canary():
    """Perfect conservation (equal inlet = outlet) should yield near-zero error."""
    metric = ConsistencyOfMass()
    flow = torch.tensor([[1.0, 2.0, 3.0]])  # [B=1, inlets=3]
    result = metric(flow, flow)
    assert result["error"].item() < 1e-5


# ---------------------------------------------------------------------------
# Parametrized
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "inlet,outlet,expect_pass",
    [
        # Perfect conservation
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[1.0, 1.0]]), True),
        # 4% error → within default 5% tolerance
        (torch.tensor([[1.0]]), torch.tensor([[0.96]]), True),
        # 50% error → fails
        (torch.tensor([[1.0]]), torch.tensor([[0.5]]), False),
    ],
)
def test_mass_conservation_pass_fail(inlet, outlet, expect_pass):
    metric = ConsistencyOfMass(tolerance=0.05)
    result = metric(inlet, outlet)
    passes = result["passes"].item()
    # passes is a float in [0,1] from the mean — check the expectation
    if expect_pass:
        assert passes > 0.5, f"Expected pass, got passes={passes}"
    else:
        assert passes < 0.5, f"Expected fail, got passes={passes}"


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_mass_conservation_batch(batch_size):
    metric = ConsistencyOfMass()
    inlets = torch.ones(batch_size, 3)
    outlets = torch.ones(batch_size, 2)
    result = metric(inlets, outlets)
    assert result["error"].ndim == 0  # scalar
    assert torch.isfinite(result["error"])


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mass_conservation_zero_flows():
    """Zero flows should not produce NaN (division guard)."""
    metric = ConsistencyOfMass()
    z = torch.zeros(1, 2)
    result = metric(z, z)
    assert torch.isfinite(result["error"])
    assert torch.isfinite(result["percent_error"])


@pytest.mark.unit
def test_divergence_metric_instantiates():
    """Divergence metric can be instantiated with custom voxel size."""
    div = Divergence(voxel_size=(1.5, 1.5, 2.0))
    assert div is not None


@pytest.mark.unit
def test_compute_from_velocity():
    """compute_from_velocity returns a scalar tensor."""
    metric = ConsistencyOfMass()
    velocity = torch.randn(1, 3, 4, 4, 4)
    plane_normal = torch.tensor([0.0, 0.0, 1.0])
    plane_mask = torch.ones(1, 1, 4, 4, 4)
    result = metric.compute_from_velocity(
        velocity, plane_normal, plane_mask, voxel_area=1.0
    )
    assert result.ndim == 0


# ---------------------------------------------------------------------------
# Expected failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mass_conservation_negative_tolerance_init():
    """Negative tolerance is allowed to construct but should behave predictably."""
    # A negative tolerance means passes would never be true (threshold < 0).
    metric = ConsistencyOfMass(tolerance=-0.1)
    inlets = torch.tensor([[1.0]])
    outlets = torch.tensor([[1.0]])
    result = metric(inlets, outlets)
    # passes = percent_error < -10 → always False
    assert result["passes"].item() == pytest.approx(0.0)
