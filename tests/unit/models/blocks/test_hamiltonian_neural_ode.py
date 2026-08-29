"""Unit tests for the Hamiltonian Neural ODE block (B-4 / Bundle P7).

Covers the load-bearing physics guarantees:
* leapfrog energy conservation to tight tolerance over the integration window,
* zero-potential consistency (straight-line / Cartesian trajectory),
* gradient flow through the learnable potential V_theta,
* deterministic seeding,
* no-NaN over a long horizon,
* the position-Verlet ("rk4_symplectic") integrator also conserves energy.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.hamiltonian_neural_ode import (
    HamiltonianNeuralODE,
    PotentialNet,
)


class _QuadraticPotential(nn.Module):
    """V(k) = 1/2 * stiffness * |k|^2 — a learnable harmonic well.

    Gives an analytically tractable, bounded, oscillatory trajectory ideal for
    testing energy conservation (a harmonic oscillator).
    """

    def __init__(self, stiffness: float = 1.0):
        super().__init__()
        self.log_k = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(stiffness)))))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        stiffness = torch.exp(self.log_k)
        return 0.5 * stiffness * (k**2).sum(dim=-1, keepdim=True)


def _make_ode(integrator="leapfrog", dt=1e-3, T_total=1.0, stiffness=4.0):  # noqa: N803
    pot = _QuadraticPotential(stiffness=stiffness)
    return HamiltonianNeuralODE(pot, integrator=integrator, dt=dt, T_total=T_total)


def test_leapfrog_conserves_energy_tightly():
    """Leapfrog energy drift must be tiny over the full window (harmonic well)."""
    torch.manual_seed(0)
    ode = _make_ode(integrator="leapfrog", dt=1e-3, T_total=1.0, stiffness=4.0)
    k0 = torch.tensor([[0.3, -0.2]])
    g0 = torch.tensor([[0.1, 0.4]])
    out = ode(k0, g0, return_energy=True)
    e = out["energy"][0]  # [T+1]
    rel_drift = (e - e[0]).abs().max() / (e[0].abs() + 1e-9)
    # Symplectic integrator: bounded oscillation, no secular drift.
    assert rel_drift.item() < 1e-3, f"energy drift too large: {rel_drift.item():.2e}"


def test_zero_potential_gives_straight_line():
    """V=const => grad V = 0 => k(t) = k0 + g0 t (Cartesian raster)."""
    torch.manual_seed(0)
    # stiffness->0 makes the potential flat (zero gradient everywhere).
    ode = _make_ode(integrator="leapfrog", dt=1e-3, T_total=0.5, stiffness=0.0)
    # Force exact zero stiffness.
    with torch.no_grad():
        ode.potential_net.log_k.fill_(float("-inf"))
    k0 = torch.tensor([[-0.5, 0.1]])
    g0 = torch.tensor([[1.0, 0.0]])
    out = ode(k0, g0)
    k = out["k"][0]  # [T+1, 2]
    t = torch.arange(k.shape[0]) * ode.dt
    expected = k0[0][None, :] + g0[0][None, :] * t[:, None]
    assert torch.allclose(k, expected, atol=1e-5), "trajectory is not a straight line"
    # Momentum constant.
    g = out["g"][0]
    assert torch.allclose(g, g0.expand_as(g), atol=1e-5)


def test_gradient_flows_through_potential():
    """A scalar of the trajectory must backprop into V_theta's parameters."""
    torch.manual_seed(0)
    ode = _make_ode(integrator="leapfrog", dt=2e-3, T_total=0.2, stiffness=3.0)
    ode.train()
    k0 = torch.tensor([[0.2, 0.1]], requires_grad=True)
    g0 = torch.tensor([[0.3, -0.1]], requires_grad=True)
    out = ode(k0, g0)
    loss = out["k"].pow(2).mean()
    loss.backward()
    grad = ode.potential_net.log_k.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert grad.abs().item() > 0.0, "no gradient reached V_theta"


def test_siren_potential_gradient_flows():
    """The real SIREN PotentialNet must also receive gradients."""
    torch.manual_seed(1)
    pot = PotentialNet(spatial_dims=2, hidden_dim=32, num_layers=2, arch="siren")
    ode = HamiltonianNeuralODE(pot, dt=2e-3, T_total=0.05)
    ode.train()
    k0 = torch.tensor([[0.1, -0.2]])
    g0 = torch.tensor([[0.5, 0.2]])
    out = ode(k0, g0, num_steps=20)
    out["k"].abs().mean().backward()
    grads = [p.grad for p in pot.parameters() if p.grad is not None]
    assert grads, "SIREN potential received no gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_deterministic_seeding():
    """Same seed + same inputs => identical trajectory."""
    def run():
        torch.manual_seed(123)
        pot = PotentialNet(spatial_dims=2, hidden_dim=16, num_layers=2, arch="mlp")
        ode = HamiltonianNeuralODE(pot, dt=1e-3, T_total=0.02)
        k0 = torch.tensor([[0.1, 0.1]])
        g0 = torch.tensor([[0.2, 0.2]])
        return ode(k0, g0, num_steps=15)["k"]

    a = run()
    b = run()
    assert torch.equal(a, b)


def test_no_nan_over_long_horizon():
    """Long integration must stay finite (stability)."""
    torch.manual_seed(0)
    pot = PotentialNet(spatial_dims=2, hidden_dim=32, num_layers=3, arch="siren")
    ode = HamiltonianNeuralODE(pot, dt=4e-6, T_total=50e-3)
    k0 = torch.tensor([[0.0, 0.0]])
    g0 = torch.tensor([[10.0, 5.0]])
    out = ode(k0, g0, num_steps=2000, return_energy=True)
    assert torch.isfinite(out["k"]).all()
    assert torch.isfinite(out["g"]).all()
    assert torch.isfinite(out["slew"]).all()
    assert torch.isfinite(out["energy"]).all()


def test_position_verlet_conserves_energy():
    """The rk4_symplectic (position-Verlet) variant is also energy-conserving."""
    torch.manual_seed(0)
    ode = _make_ode(integrator="rk4_symplectic", dt=1e-3, T_total=1.0, stiffness=4.0)
    k0 = torch.tensor([[0.25, -0.1]])
    g0 = torch.tensor([[0.0, 0.3]])
    e = ode(k0, g0, return_energy=True)["energy"][0]
    rel_drift = (e - e[0]).abs().max() / (e[0].abs() + 1e-9)
    assert rel_drift.item() < 1e-3


def test_shapes_and_slew_definition():
    """Output shapes and slew = -grad V at knots."""
    torch.manual_seed(0)
    pot = _QuadraticPotential(stiffness=2.0)
    ode = HamiltonianNeuralODE(pot, dt=1e-3, T_total=0.01)
    k0 = torch.tensor([[0.1, 0.2], [0.0, -0.1]])  # batch of 2 shots
    g0 = torch.tensor([[0.3, 0.0], [0.1, 0.1]])
    out = ode(k0, g0, num_steps=10)
    assert out["k"].shape == (2, 11, 2)
    assert out["g"].shape == (2, 11, 2)
    assert out["slew"].shape == (2, 11, 2)
    # slew at t=0 must equal -grad V(k0) = -stiffness * k0.
    expected_slew0 = -2.0 * k0
    assert torch.allclose(out["slew"][:, 0], expected_slew0, atol=1e-5)


def test_potential_arch_rejects_unknown():
    """Unknown potential arch raises (no silent fallback, CLAUDE.md #9)."""
    import pytest

    with pytest.raises(ValueError, match="Unknown potential arch"):
        PotentialNet(arch="quantum")  # type: ignore[arg-type]


def test_integrator_rejects_unknown():
    """Unknown integrator raises loudly."""
    import pytest

    pot = _QuadraticPotential()
    with pytest.raises(ValueError, match="Unknown integrator"):
        HamiltonianNeuralODE(pot, integrator="symplectic_magic")  # type: ignore[arg-type]
