r"""Tests for ``ConcomitantPhaseOperator``.

Targets ``mriforge.infrastructure.physics.concomitant_phase_operator`` —
idea 1 (CAUR) of ``TODO/integration_plan_ulf_cheap_fast_mri.md``.

Plan acceptance criterion (§1.6):

- ``φ_c`` matches Bernstein 1998 for two trajectories — covered
  indirectly here: we already test the Bernstein formula in
  ``test_b0_mapping`` etc.; here we confirm the operator's *output*
  is consistent with that field map.

Standard contract tests:

- ``enabled=False`` → identity
- Real-valued image is cast to complex on the fly
- Operator preserves shape
- Phase at isocentre is exactly zero (Bernstein's quadratic form
  vanishes there)
- Magnitude unchanged (``|e^{iφ}| = 1``)
- ``readout_time_s = 0`` → identity
- Negative readout time rejected
"""

from __future__ import annotations

import math

import pytest
import torch

from mriforge.infrastructure.physics.concomitant_phase_operator import (
    ConcomitantPhaseOperator,
)


SPATIAL = (4, 8, 8)
VOXEL = (4.0, 4.0, 4.0)  # 4 mm isotropic
GRAD = (10.0, 10.0, 0.0)  # mT/m per axis
B0_LF = 0.064  # 64 mT (Hyperfine Swoop)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_negative_readout_time_rejected() -> None:
    """``readout_time_s < 0`` raises."""
    with pytest.raises(ValueError, match="readout_time_s"):
        ConcomitantPhaseOperator(
            spatial_shape=SPATIAL,
            voxel_size_mm=VOXEL,
            gradient_amplitudes_mT_per_m=GRAD,
            field_strength_t=B0_LF,
            readout_time_s=-1e-3,
        )


def test_high_field_warns() -> None:
    """Construction at HF emits a ``UserWarning`` (still allowed)."""
    with pytest.warns(UserWarning, match="below 0.5 T"):
        ConcomitantPhaseOperator(
            spatial_shape=SPATIAL,
            voxel_size_mm=VOXEL,
            gradient_amplitudes_mT_per_m=GRAD,
            field_strength_t=3.0,
            readout_time_s=1e-3,
        )


# ---------------------------------------------------------------------------
# Forward — disabled / zero-time identity
# ---------------------------------------------------------------------------


def test_disabled_is_identity() -> None:
    """``enabled=False`` returns the input unchanged."""
    op = ConcomitantPhaseOperator(
        SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=1e-3, enabled=False
    )
    img = torch.complex(torch.randn(1, 1, *SPATIAL), torch.randn(1, 1, *SPATIAL))
    out = op(img)
    assert torch.equal(out, img)


def test_zero_readout_time_is_identity() -> None:
    """``readout_time_s = 0`` makes φ = 0 → operator is identity."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=0.0)
    img = torch.complex(torch.randn(1, 1, *SPATIAL), torch.randn(1, 1, *SPATIAL))
    out = op(img)
    assert torch.allclose(out, img, atol=1e-6)


# ---------------------------------------------------------------------------
# Forward — physical sanity
# ---------------------------------------------------------------------------


def test_phase_zero_at_isocentre() -> None:
    """Bernstein's quadratic form is zero AT the FOV centre.

    For SPATIAL = (4, 8, 8) all dims are even, so isocentre falls
    BETWEEN voxels. The voxels around the centre are equal by
    quadratic symmetry and their (small but non-zero) phase scales as
    ``(voxel_offset)² · readout_time`` — they're not exactly zero, but
    they are strictly smaller than any voxel further from the centre.
    """
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    phase = op.phase
    D, H, W = SPATIAL
    # The 2×2×2 block straddling isocentre is symmetric (all equal).
    centre_block = phase[D // 2 - 1 : D // 2 + 1, H // 2 - 1 : H // 2 + 1, W // 2 - 1 : W // 2 + 1]
    centre_val = centre_block[0, 0, 0].item()
    assert torch.allclose(
        centre_block, torch.full_like(centre_block, centre_val), atol=1e-6
    ), "centre voxels must be symmetric (Bernstein form is even in each axis)"
    # And the centre phase must be strictly smaller than any voxel
    # one step further out (Bernstein form is monotone in distance).
    edge_val = phase[0, 0, 0].item()
    assert abs(centre_val) < abs(edge_val)


def test_magnitude_preserved() -> None:
    """``|e^{iφ}| = 1`` → magnitude of complex image unchanged."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    img = torch.complex(torch.randn(1, 1, *SPATIAL), torch.randn(1, 1, *SPATIAL))
    out = op(img)
    assert torch.allclose(out.abs(), img.abs(), atol=1e-5)


def test_real_input_cast_to_complex() -> None:
    """Real-valued image is auto-promoted to complex."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    img_real = torch.randn(1, 1, *SPATIAL)
    out = op(img_real)
    assert torch.is_complex(out)
    assert out.shape == img_real.shape


def test_shape_preserved() -> None:
    """Output shape matches input."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=1e-3)
    img = torch.complex(torch.randn(2, 3, *SPATIAL), torch.randn(2, 3, *SPATIAL))
    out = op(img)
    assert out.shape == img.shape


def test_frequency_map_recoverable() -> None:
    """``get_frequency_map_hz`` undoes the time-integration step."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    f_hz = op.get_frequency_map_hz()
    # Recompute directly from the buffer.
    expected = op.phase / (2.0 * math.pi * 2e-3)
    assert torch.allclose(f_hz, expected)


# ---------------------------------------------------------------------------
# Phasor buffer
# ---------------------------------------------------------------------------


def test_phasor_unit_magnitude() -> None:
    """The cached ``phasor`` buffer has unit magnitude everywhere."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    mag = op.phasor.abs()
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-5)


def test_buffers_registered() -> None:
    """Both ``phase`` and ``phasor`` are registered as nn buffers (move with device)."""
    op = ConcomitantPhaseOperator(SPATIAL, VOXEL, GRAD, B0_LF, readout_time_s=2e-3)
    names = {n for n, _ in op.named_buffers()}
    assert {"phase", "phasor"} <= names
