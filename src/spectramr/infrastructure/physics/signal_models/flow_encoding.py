"""Phase-contrast (PC) / 4D-flow signal model.

Phase-contrast MRI encodes velocity in the *phase* of the complex signal via a
bipolar gradient. With a chosen velocity-encoding value ``venc`` the phase is

.. math::

    \\phi = \\pi \\, v / v_{enc}

so a spin moving at ``v = venc`` accrues exactly :math:`\\pi` of phase and
``|v| > venc`` aliases (phase wraps). These are pure, differentiable tensor
ops with an exact analytic round-trip (``v -> phi -> v`` at known ``venc``),
which is what the physics tests pin.

The gradient first moment realising a given ``venc`` is
:math:`M_1 = \\pi / (\\gamma\\, v_{enc})` with ``gamma`` the gyromagnetic ratio
in rad·s⁻¹·T⁻¹ (¹H: ``2.675e8``).
"""

from __future__ import annotations

import math

import torch

#: Gyromagnetic ratio of ¹H, rad·s⁻¹·T⁻¹.
GAMMA_H: float = 2.6752218744e8


def velocity_to_phase(velocity: torch.Tensor, venc: float) -> torch.Tensor:
    """Map velocity to encoded phase: ``phi = pi * v / venc`` (radians)."""
    if venc <= 0:
        raise ValueError(f"venc must be positive, got {venc}")
    return math.pi * velocity / venc


def phase_to_velocity(phase: torch.Tensor, venc: float) -> torch.Tensor:
    """Map encoded phase back to velocity: ``v = phi * venc / pi``."""
    if venc <= 0:
        raise ValueError(f"venc must be positive, got {venc}")
    return phase * venc / math.pi


def venc_to_first_moment(venc: float, gamma: float = GAMMA_H) -> float:
    """Gradient first moment ``M1 = pi / (gamma * venc)`` for a target ``venc``."""
    if venc <= 0:
        raise ValueError(f"venc must be positive, got {venc}")
    return math.pi / (gamma * venc)


def pc_forward(
    velocity: torch.Tensor,
    venc: float,
    magnitude: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Forward PC model: velocity -> complex flow-encoded signal.

    ``signal = magnitude * exp(i * pi * v / venc)``.
    """
    phase = velocity_to_phase(velocity, venc)
    mag = (
        magnitude
        if isinstance(magnitude, torch.Tensor)
        else torch.as_tensor(magnitude, dtype=phase.dtype, device=phase.device)
    )
    return mag * torch.exp(1j * phase.to(torch.complex64))


def pc_adjoint(signal: torch.Tensor, venc: float) -> torch.Tensor:
    """Adjoint PC model: complex signal -> velocity (via the wrapped phase)."""
    return phase_to_velocity(torch.angle(signal), venc)


def four_point_reference_encode(velocity: torch.Tensor, venc: float) -> torch.Tensor:
    """Simple 4-point (reference + 3 directional) phase encode.

    Args:
        velocity: ``[..., 3]`` (vx, vy, vz).

    Returns:
        ``[..., 4]`` phases (reference, x, y, z). The reference carries no
        flow encoding; each directional phase is the reference plus that
        component's encoded phase.
    """
    if velocity.shape[-1] != 3:
        raise ValueError(f"expected last dim 3 (vx,vy,vz), got {velocity.shape}")
    ref_shape = (*velocity.shape[:-1], 1)
    ref = torch.zeros(ref_shape, dtype=velocity.dtype, device=velocity.device)
    directional = velocity_to_phase(velocity, venc)  # [..., 3]
    return torch.cat([ref, ref + directional], dim=-1)  # [..., 4]


def four_point_reference_decode(phases: torch.Tensor, venc: float) -> torch.Tensor:
    """Inverse of :func:`four_point_reference_encode`.

    Args:
        phases: ``[..., 4]`` (reference, x, y, z).

    Returns:
        ``[..., 3]`` velocity (vx, vy, vz).
    """
    if phases.shape[-1] != 4:
        raise ValueError(f"expected last dim 4, got {phases.shape}")
    ref = phases[..., :1]
    directional = phases[..., 1:] - ref  # [..., 3]
    return phase_to_velocity(directional, venc)


__all__ = [
    "GAMMA_H",
    "four_point_reference_decode",
    "four_point_reference_encode",
    "pc_adjoint",
    "pc_forward",
    "phase_to_velocity",
    "velocity_to_phase",
    "venc_to_first_moment",
]
