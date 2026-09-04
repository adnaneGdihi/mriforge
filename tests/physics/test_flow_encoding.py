"""Analytic round-trip tests for the phase-contrast / 4D-flow signal model."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.signal_models.flow_encoding import (
    four_point_reference_decode,
    four_point_reference_encode,
    pc_adjoint,
    pc_forward,
    phase_to_velocity,
    velocity_to_phase,
    venc_to_first_moment,
)


def test_velocity_phase_roundtrip() -> None:
    venc = 150.0
    v = torch.linspace(-venc, venc, 101)
    recovered = phase_to_velocity(velocity_to_phase(v, venc), venc)
    assert torch.allclose(recovered, v, atol=1e-4)


def test_phase_is_pi_at_venc() -> None:
    venc = 80.0
    phi = velocity_to_phase(torch.tensor(venc), venc)
    assert math.isclose(phi.item(), math.pi, rel_tol=1e-6)


def test_pc_forward_adjoint_roundtrip_within_venc() -> None:
    venc = 100.0
    # Stay strictly inside (-venc, venc) so angle() does not wrap.
    v = torch.linspace(-venc * 0.9, venc * 0.9, 50)
    sig = pc_forward(v, venc, magnitude=2.0)
    assert torch.allclose(sig.abs(), torch.full_like(v, 2.0), atol=1e-4)
    v_rec = pc_adjoint(sig, venc)
    assert torch.allclose(v_rec, v, atol=1e-3)


def test_pc_aliases_beyond_venc() -> None:
    venc = 50.0
    v = torch.tensor(1.5 * venc)  # exceeds venc -> should wrap
    v_rec = pc_adjoint(pc_forward(v, venc), venc)
    assert not torch.allclose(v_rec, v, atol=1.0)  # aliased


def test_four_point_reference_roundtrip() -> None:
    venc = 120.0
    v = torch.randn(4, 3) * (venc * 0.3)
    phases = four_point_reference_encode(v, venc)
    assert phases.shape == (4, 4)
    v_rec = four_point_reference_decode(phases, venc)
    assert torch.allclose(v_rec, v, atol=1e-3)


def test_venc_to_first_moment_positive() -> None:
    m1 = venc_to_first_moment(100.0)
    assert m1 > 0
    # M1 scales inversely with venc.
    assert venc_to_first_moment(50.0) > m1


def test_invalid_venc_raises() -> None:
    with pytest.raises(ValueError):
        velocity_to_phase(torch.tensor(1.0), 0.0)


def test_operator_registered_and_roundtrips() -> None:
    from spectramr.infrastructure.physics.registry import (
        create_operator,
        list_operators,
    )

    assert "phase_contrast" in list_operators()
    op = create_operator("phase_contrast", venc=100.0)
    v = torch.linspace(-90.0, 90.0, 40)
    v_rec = op.adjoint(op.forward(v))
    assert torch.allclose(v_rec, v, atol=1e-3)
    assert op.get_operator_type() == "phase_contrast"
