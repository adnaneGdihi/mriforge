"""Unit tests for the Hamiltonian trajectory generator model (B-4 / Bundle P7)."""

from __future__ import annotations

import pytest
import torch

from mriforge.models.acquisition.hamiltonian_trajectory_generator import (
    HamiltonianTrajectoryGenerator,
)
from mriforge.models.registry import MODEL_REGISTRY, get_model_class


def test_registered_in_model_registry():
    """The @register_model decorator must have fired with the right metadata."""
    assert "hamiltonian_trajectory_generator" in MODEL_REGISTRY
    entry = MODEL_REGISTRY["hamiltonian_trajectory_generator"]
    assert entry["mode"] == "acquisition_learning"
    assert get_model_class("hamiltonian_trajectory_generator") is HamiltonianTrajectoryGenerator
    caps = entry["capabilities"]
    assert caps.accepts_complex is False
    # I/O contract matches the sibling learnable-acquisition models (loupe/pilot):
    # k-space target in, image recon out. Trajectory waveforms are physics
    # internals, not the model's declared data I/O domain.
    assert caps.input_domain == "kspace"
    assert caps.output_domain == "image"


def test_forward_default_initial_conditions():
    """Forward with no ICs uses the Cartesian-raster init and returns shapes."""
    torch.manual_seed(0)
    gen = HamiltonianTrajectoryGenerator(
        spatial_dims=2, potential_hidden_dim=16, potential_num_layers=2,
        num_shots=4, samples_per_shot=32,
    )
    out = gen()
    assert out["k"].shape == (4, 32, 2)
    assert out["g"].shape == (4, 32, 2)
    assert out["slew"].shape == (4, 32, 2)
    assert out["energy"].shape == (4, 32)
    assert torch.isfinite(out["k"]).all()


def test_zero_potential_init_is_cartesian():
    """A freshly-zeroed SIREN-final-layer potential yields a near-straight raster.

    We zero the final linear layer so V_theta ~ const => grad ~ 0 => straight
    lines from the Cartesian initial conditions.
    """
    torch.manual_seed(0)
    gen = HamiltonianTrajectoryGenerator(
        spatial_dims=2, potential_arch="mlp", potential_hidden_dim=16,
        potential_num_layers=2, num_shots=8, samples_per_shot=16,
    )
    # Zero the output layer => potential is a constant => zero gradient field.
    last = gen.potential_net.net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    out = gen()
    k = out["k"]
    # Readout axis should sweep monotonically from -0.5 upward for every shot.
    sweep = k[:, -1, 0] - k[:, 0, 0]
    assert (sweep > 0).all()
    # Phase-encode axis stays put (no force).
    pe_drift = (k[:, -1, 1] - k[:, 0, 1]).abs().max()
    assert pe_drift.item() < 1e-4


def test_explicit_initial_conditions_respected():
    """Passed (k0, g0) override the defaults."""
    torch.manual_seed(0)
    gen = HamiltonianTrajectoryGenerator(
        spatial_dims=2, potential_hidden_dim=16, potential_num_layers=2,
        num_shots=2, samples_per_shot=8,
    )
    k0 = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
    g0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = gen(k0, g0)
    assert torch.allclose(out["k"][:, 0], k0, atol=1e-5)
    assert torch.allclose(out["g"][:, 0], g0, atol=1e-5)


def test_gradient_flows_to_potential():
    """Backprop from trajectory reaches the model's parameters."""
    torch.manual_seed(0)
    gen = HamiltonianTrajectoryGenerator(
        spatial_dims=2, potential_arch="siren", potential_hidden_dim=16,
        potential_num_layers=2, num_shots=2, samples_per_shot=16,
    )
    gen.train()
    out = gen()
    out["k"].abs().mean().backward()
    grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        HamiltonianTrajectoryGenerator(num_shots=0)
    with pytest.raises(ValueError):
        HamiltonianTrajectoryGenerator(samples_per_shot=1)
