"""Unit tests for src/spectramr/models/physics/therm_pinn.py (moved 2026-07-02
from infrastructure/physics/ — it is a registered model, not a physics
primitive).

Coverage for the (previously unreferenced) Therm-PINN bio-heat module:
  - ``TemperatureNetwork`` / ``ThermPINN.forward`` shape + finiteness
  - ``compute_pde_residual`` (autograd dT/dt + Laplacian) shape + finiteness
  - ``compute_loss`` returns a finite scalar + a dict with the documented keys
  - safety penalty is ~0 at the default body-temperature bias (37 < T_limit=40)
  - ``create_therm_pinn`` factory returns a ``ThermPINN``

NOTE: ``predict_max_safe_SAR`` is a FLAGGED design bug — ``T_net`` does not take
SAR as input, so the SAR sweep is meaningless. We therefore only assert that it
runs and returns a scalar tensor; we do NOT assert the sweep is correct.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.physics.therm_pinn import (
    TemperatureNetwork,
    ThermPINN,
    create_therm_pinn,
)


@pytest.mark.physics
def test_temperature_network_forward_shape() -> None:
    """TemperatureNetwork(coords[B,3], t[B,1]) -> T[B,1], finite."""
    torch.manual_seed(0)
    net = TemperatureNetwork(hidden_dims=(16, 16))
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    T = net(coords, t)
    assert T.shape == (8, 1)
    assert torch.isfinite(T).all()


@pytest.mark.physics
def test_therm_pinn_forward_shape() -> None:
    """ThermPINN.forward returns [B, 1] and at init predicts ~body temperature."""
    torch.manual_seed(0)
    model = ThermPINN(hidden_dims=(16, 16))
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    T = model(coords, t)
    assert T.shape == (8, 1)
    assert torch.isfinite(T).all()
    # Output layer is zero-init weight + bias 37.0 -> every prediction == 37.0.
    assert torch.allclose(T, torch.full_like(T, 37.0), atol=1e-4)


@pytest.mark.physics
def test_compute_pde_residual_shape_and_finite() -> None:
    """compute_pde_residual uses autograd for dT/dt + Laplacian; shape [B,1]."""
    torch.manual_seed(0)
    model = ThermPINN(hidden_dims=(16, 16))
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    sar = torch.rand(8, 1)
    residual = model.compute_pde_residual(coords, t, sar)
    assert residual.shape == (8, 1)
    assert torch.isfinite(residual).all()


@pytest.mark.physics
def test_compute_loss_keys_and_finite() -> None:
    """compute_loss returns (finite scalar total, dict with pde/safety/total)."""
    torch.manual_seed(0)
    model = ThermPINN(hidden_dims=(16, 16))
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    sar = torch.rand(8, 1)
    total, loss_dict = model.compute_loss(coords, t, sar)

    assert torch.is_tensor(total)
    assert total.ndim == 0
    assert torch.isfinite(total)

    for key in ("pde", "safety", "total"):
        assert key in loss_dict, f"missing loss key {key!r}"
        assert torch.isfinite(torch.tensor(loss_dict[key]))


@pytest.mark.physics
def test_safety_penalty_zero_at_body_temperature() -> None:
    """At init the network predicts ~37 C (< T_limit=40 C), so the safety
    penalty (max(0, T - T_limit)^2) must be ~0."""
    torch.manual_seed(0)
    model = ThermPINN(hidden_dims=(16, 16), T_limit=40.0)
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    sar = torch.rand(8, 1)
    _, loss_dict = model.compute_loss(coords, t, sar)
    assert loss_dict["safety"] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.physics
def test_safety_penalty_positive_when_exceeding_limit() -> None:
    """Forcing the predicted temperature above T_limit yields a positive safety
    penalty (the F.relu(T - T_limit)^2 branch fires)."""
    torch.manual_seed(0)
    # Set the limit BELOW body temperature so the default ~37 C predictions
    # violate it -> positive penalty (exercises the violation path).
    model = ThermPINN(hidden_dims=(16, 16), T_limit=10.0)
    coords = torch.randn(8, 3)
    t = torch.randn(8, 1)
    sar = torch.rand(8, 1)
    _, loss_dict = model.compute_loss(coords, t, sar)
    assert loss_dict["safety"] > 0.0


@pytest.mark.physics
def test_predict_max_safe_sar_runs_and_returns_scalar() -> None:
    """predict_max_safe_SAR is a FLAGGED design bug (T_net ignores SAR), so we
    only assert it runs and returns a finite scalar — NOT that the sweep is
    physically meaningful."""
    torch.manual_seed(0)
    model = ThermPINN(hidden_dims=(16, 16))
    coords = torch.randn(1, 3)
    t = torch.randn(1, 1)
    out = model.predict_max_safe_SAR(coords, t, SAR_range=(0.1, 5.0), num_samples=8)
    assert torch.is_tensor(out)
    assert out.ndim == 0
    assert torch.isfinite(out)


@pytest.mark.physics
def test_create_therm_pinn_factory() -> None:
    """create_therm_pinn returns a configured ThermPINN with the given T_limit."""
    model = create_therm_pinn(hidden_dims=(8, 8), T_limit=42.0)
    assert isinstance(model, ThermPINN)
    assert float(model.T_limit.item()) == pytest.approx(42.0)
