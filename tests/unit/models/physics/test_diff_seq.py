"""Tests for ``BlochSimulator``, ``LearnableSequence``, and ``DiffSEQ``.

Targets ``mriforge.models.physics.diff_seq`` (moved 2026-07-02 from
``infrastructure/physics/`` — it is a registered model, not a physics
primitive) — differentiable pulse
sequence simulator. Heavy parameter-recovery tests are deferred to
integration; here we verify construction, parameter constraints, and
gradient flow through the building blocks.
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.models.physics.diff_seq import (
    BlochSimulator,
    DiffSEQ,
    LearnableSequence,
    create_diff_seq,
)


# ---------------------------------------------------------------------------
# BlochSimulator construction
# ---------------------------------------------------------------------------


def test_bloch_simulator_persists_constants() -> None:
    """Constructor stores T1, T2, M0, dt as buffers / attributes."""
    sim = BlochSimulator(T1=800.0, T2=80.0, M0=0.5, dt=0.05)
    assert sim.T1.item() == 800.0
    assert sim.T2.item() == 80.0
    assert sim.M0.item() == 0.5
    assert sim.dt == 0.05


def test_bloch_constants_registered_as_buffers() -> None:
    """T1/T2/M0 are buffers (move with `.to(device)`)."""
    sim = BlochSimulator()
    buffer_names = {n for n, _ in sim.named_buffers()}
    assert {"T1", "T2", "M0"} <= buffer_names


# ---------------------------------------------------------------------------
# LearnableSequence
# ---------------------------------------------------------------------------


def test_learnable_sequence_default_persists_pulse_count() -> None:
    """Constructor stores num_pulses worth of parameters."""
    seq = LearnableSequence(num_pulses=64, init_flip_angle=15.0, init_TR=20.0)
    assert seq.flip_angle_logit.shape == (64,)
    assert seq.phase.shape == (64,)
    assert seq.TR_log.shape == (64,)


def test_get_flip_angles_in_zero_pi_range() -> None:
    """Flip angles are sigmoid-bounded to ``[0, π]``."""
    seq = LearnableSequence(num_pulses=10, init_flip_angle=10.0)
    angles = seq.get_flip_angles()
    assert (angles >= 0).all()
    assert (angles <= math.pi).all()


def test_get_TRs_strictly_positive() -> None:
    """TRs are exp-transformed → always > 0."""
    seq = LearnableSequence(num_pulses=10, init_TR=15.0)
    TRs = seq.get_TRs()
    assert (TRs > 0).all()


def test_compute_SAR_sums_squared_flip_angles() -> None:
    """SAR is ``sum(flip_angles²)``."""
    seq = LearnableSequence(num_pulses=10, init_flip_angle=10.0)
    sar = seq.compute_SAR()
    expected = (seq.get_flip_angles() ** 2).sum()
    assert torch.allclose(sar, expected)


def test_init_flip_angle_round_trips() -> None:
    """``init_flip_angle=10°`` → ``get_flip_angles()`` ≈ 10° in radians."""
    seq = LearnableSequence(num_pulses=4, init_flip_angle=10.0)
    angles = seq.get_flip_angles()
    expected_rad = 10.0 * math.pi / 180.0
    # Sigmoid-mapped, so allow some tolerance.
    assert torch.allclose(
        angles, torch.full_like(angles, expected_rad), atol=1e-2
    )


def test_init_TR_round_trips() -> None:
    """``init_TR=10`` → ``get_TRs()`` ≈ 10."""
    seq = LearnableSequence(num_pulses=4, init_TR=10.0)
    TRs = seq.get_TRs()
    assert torch.allclose(TRs, torch.full_like(TRs, 10.0), atol=1e-3)


def test_learnable_sequence_parameters_have_gradient() -> None:
    """All three parameter tensors are learnable."""
    seq = LearnableSequence(num_pulses=4)
    assert seq.flip_angle_logit.requires_grad is True
    assert seq.phase.requires_grad is True
    assert seq.TR_log.requires_grad is True


# ---------------------------------------------------------------------------
# DiffSEQ — top-level
# ---------------------------------------------------------------------------


def test_diff_seq_construction_with_default_recon_net() -> None:
    """Without explicit recon_net, a default MLP is created."""
    model = DiffSEQ(num_pulses=10)
    assert isinstance(model.recon_net, torch.nn.Sequential)


def test_diff_seq_construction_with_custom_recon_net() -> None:
    """Custom recon_net replaces the default."""
    custom = torch.nn.Linear(20, 2)
    model = DiffSEQ(num_pulses=10, recon_net=custom)
    assert model.recon_net is custom


def test_diff_seq_persists_sar_weight() -> None:
    """``sar_weight`` is stored on the instance."""
    model = DiffSEQ(num_pulses=8, sar_weight=0.05)
    assert model.sar_weight == 0.05


# ---------------------------------------------------------------------------
# create_diff_seq factory
# ---------------------------------------------------------------------------


def test_create_diff_seq_factory_returns_diffseq() -> None:
    """Factory returns a fully-configured DiffSEQ."""
    model = create_diff_seq(num_pulses=8, T1=900.0, T2=90.0)
    assert isinstance(model, DiffSEQ)


def test_create_diff_seq_propagates_T1_T2() -> None:
    """T1/T2 are forwarded to the internal Bloch simulator."""
    model = create_diff_seq(num_pulses=8, T1=900.0, T2=90.0)
    assert model.simulator.T1.item() == 900.0
    assert model.simulator.T2.item() == 90.0


# ---------------------------------------------------------------------------
# Regression: RF rotation handedness (2026-07)
# ---------------------------------------------------------------------------
# The BlochSimulator RF block previously used the transpose of the Rodrigues
# matrix (rotation by -flip_angle), so a 90°_x pulse on equilibrium produced
# (0, +1, 0) instead of the physical (0, -1, 0) — opposite handedness to the
# free-precession block. The signs now match standard Rodrigues.


def _no_relax_sim() -> BlochSimulator:
    # dt=0 with huge T1/T2 → relaxation factors ≈ 1 and precession angle 0.
    return BlochSimulator(T1=1e9, T2=1e9, dt=0.0)


def test_bloch_90x_rotates_equilibrium_to_minus_y() -> None:
    """90°_x on (0,0,1) → (0,-1,0) (right-handed rotation about +x)."""
    out = _no_relax_sim()(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([math.pi / 2]),
        torch.tensor([0.0]),  # phase 0 → axis +x
    )
    assert torch.allclose(out[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5)


def test_bloch_90y_rotates_equilibrium_to_plus_x() -> None:
    """90°_y on (0,0,1) → (+1,0,0) (right-handed rotation about +y)."""
    out = _no_relax_sim()(
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([math.pi / 2]),
        torch.tensor([math.pi / 2]),  # phase π/2 → axis +y
    )
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
