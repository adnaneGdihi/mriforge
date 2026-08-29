"""Tests for ``ConcomitantAwareMixin``.

Targets ``mriforge.infrastructure.training.strategies.mixins.concomitant_aware_mixin``.

CAUR (idea 1) phase 2 — wires the previously-built
``ConcomitantPhaseOperator`` into a host training strategy via
cooperative MRO.

Categories:

- Defaults to *disabled*; ``apply_concomitant_correction`` is a no-op
- Accessing ``concomitant_op`` before configuration raises
- After ``configure_concomitant_correction``, the operator is active
- ``apply_concomitant_correction`` preserves shape and runs the
  operator in the same way the standalone class does
- Magnitude is preserved under the correction (``|e^{iφ}| = 1``)
- HF-warning bubbles up from the underlying operator
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.training.strategies.mixins.concomitant_aware_mixin import (
    ConcomitantAwareMixin,
)


SPATIAL = (4, 8, 8)
VOXEL = (4.0, 4.0, 4.0)
GRAD = (10.0, 10.0, 0.0)
B0_LF = 0.064


class _Host(ConcomitantAwareMixin):
    """Minimal host class to exercise the mixin in isolation."""


# ---------------------------------------------------------------------------
# Disabled / configure contract
# ---------------------------------------------------------------------------


def test_default_disabled() -> None:
    """A fresh host has the correction disabled."""
    host = _Host()
    assert host.concomitant_enabled is False


def test_apply_when_disabled_is_identity() -> None:
    """Without configuration, ``apply_concomitant_correction`` is a no-op."""
    host = _Host()
    img = torch.complex(torch.randn(1, 1, *SPATIAL), torch.randn(1, 1, *SPATIAL))
    out = host.apply_concomitant_correction(img)
    assert torch.equal(out, img)


def test_concomitant_op_property_raises_before_configure() -> None:
    """Accessing the underlying operator before ``configure_*`` raises."""
    host = _Host()
    with pytest.raises(RuntimeError, match="not configured"):
        _ = host.concomitant_op


# ---------------------------------------------------------------------------
# Configured behaviour
# ---------------------------------------------------------------------------


def test_configured_mixin_enables_op() -> None:
    """After configuration, the mixin reports enabled and exposes the op."""
    host = _Host()
    host.configure_concomitant_correction(
        spatial_shape=SPATIAL,
        voxel_size_mm=VOXEL,
        gradient_amplitudes_mT_per_m=GRAD,
        field_strength_t=B0_LF,
        readout_time_s=2e-3,
    )
    assert host.concomitant_enabled is True
    assert host.concomitant_op.readout_time_s == 2e-3


def test_apply_preserves_shape() -> None:
    """``apply_concomitant_correction`` preserves the input shape."""
    host = _Host()
    host.configure_concomitant_correction(
        spatial_shape=SPATIAL,
        voxel_size_mm=VOXEL,
        gradient_amplitudes_mT_per_m=GRAD,
        field_strength_t=B0_LF,
        readout_time_s=2e-3,
    )
    img = torch.complex(torch.randn(2, 3, *SPATIAL), torch.randn(2, 3, *SPATIAL))
    out = host.apply_concomitant_correction(img)
    assert out.shape == img.shape


def test_apply_preserves_magnitude() -> None:
    """``|e^{iφ_c}| = 1`` so |out| = |in|."""
    host = _Host()
    host.configure_concomitant_correction(
        spatial_shape=SPATIAL,
        voxel_size_mm=VOXEL,
        gradient_amplitudes_mT_per_m=GRAD,
        field_strength_t=B0_LF,
        readout_time_s=2e-3,
    )
    img = torch.complex(torch.randn(1, 1, *SPATIAL), torch.randn(1, 1, *SPATIAL))
    out = host.apply_concomitant_correction(img)
    assert torch.allclose(out.abs(), img.abs(), atol=1e-5)


def test_real_input_promoted_to_complex() -> None:
    """A real-valued image is auto-promoted to complex inside the op."""
    host = _Host()
    host.configure_concomitant_correction(
        spatial_shape=SPATIAL,
        voxel_size_mm=VOXEL,
        gradient_amplitudes_mT_per_m=GRAD,
        field_strength_t=B0_LF,
        readout_time_s=2e-3,
    )
    img = torch.randn(1, 1, *SPATIAL)
    out = host.apply_concomitant_correction(img)
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# HF warning bubbles up from the underlying operator
# ---------------------------------------------------------------------------


def test_high_field_configuration_warns() -> None:
    """Configuring at HF emits the operator's UserWarning."""
    host = _Host()
    with pytest.warns(UserWarning, match="below 0.5 T"):
        host.configure_concomitant_correction(
            spatial_shape=SPATIAL,
            voxel_size_mm=VOXEL,
            gradient_amplitudes_mT_per_m=GRAD,
            field_strength_t=3.0,
            readout_time_s=2e-3,
        )
