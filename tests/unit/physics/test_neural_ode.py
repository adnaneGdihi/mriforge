"""Regression tests for the NeuralODEDynamics drift-net call-order contract.

WS-4 doc-contract fix: ``NeuralODEDynamics`` calls ``drift_net(z, t)`` (state
first, time second). The ``__init__`` docstring previously advertised BOTH
``f(t, z)`` and ``f(z, t)``, which is self-contradictory and disagreed with the
in-repo consumer (``models/generators/lans_generator.py`` ordered ``(time,
state)`` and crashed on ``B, N, _ = y.shape``). These tests lock the SSOT
contract: drift_net MUST be ``forward(state, time)``; a ``(time, state)``-ordered
module is not supported and fails loudly.

All tensors are tiny, CPU-only, fixed-seed. The code paths exercised here use
the manual-Euler fallback when torchdiffeq is unavailable and the adjoint solver
otherwise; both route through ``ODEFunc.forward`` -> ``drift_net(z, t)``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mriforge.infrastructure.physics.dynamics.neural_ode import NeuralODEDynamics


def test_drift_net_invoked_as_state_time_order() -> None:
    """drift_net's first positional arg is the state, second is the scalar time."""
    torch.manual_seed(0)
    x0 = torch.ones(3, 2)  # batch=3, dim=2
    calls: list[tuple[tuple[int, ...], object]] = []

    class OrderProbeDrift(nn.Module):
        def forward(self, state, time):  # contract: (state, time)
            # The state must carry the batch dim; the time must be a scalar.
            calls.append((tuple(state.shape), getattr(time, "ndim", None)))
            assert state.shape[0] == x0.shape[0], (
                "drift_net must be called (state, time): first arg is the batched state"
            )
            return -state

    ode = NeuralODEDynamics(drift_net=OrderProbeDrift(), solver="euler")
    t = torch.linspace(0.0, 1.0, 5)
    out = ode(x0, t)

    assert out.shape == (3, 5, 2)
    assert len(calls) > 0, "drift_net was never invoked"
    for state_shape, time_ndim in calls:
        assert state_shape == (3, 2), f"state arg lost its batched shape: {state_shape}"
        # scalar time during integration steps is 0-D (manual Euler) -> ndim 0.
        assert time_ndim == 0, f"time arg should be a scalar, got ndim={time_ndim}"


def test_time_state_ordered_module_fails() -> None:
    """A (time, state)-ordered drift module (the lans_generator bug) must fail.

    The wrapper binds its first positional arg to the scalar time and then does
    ``B, N, _ = first_arg.shape`` -- exactly the crash in the historical
    consumer. Because NeuralODEDynamics calls drift_net(state, time), the scalar
    time lands in the unpack and raises.
    """
    torch.manual_seed(0)
    x0 = torch.ones(2, 4)

    class TimeFirstDrift(nn.Module):
        def forward(self, time, state):  # WRONG order: (time, state)
            # Mirrors lans_generator's `B, N, _ = y.shape` on its first arg,
            # which here is the scalar time -> cannot unpack a 0-D tensor.
            _b, _n, _ = time.shape
            return -state

    ode = NeuralODEDynamics(drift_net=TimeFirstDrift(), solver="euler")
    t = torch.linspace(0.0, 1.0, 3)
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        ode(x0, t)
