r"""Symplectic Munthe-Kaas integrator for the Bloch equation on SO(3).

The Bloch equation without relaxation,

.. math::

    \dot{\mathbf{M}} = \gamma\,\mathbf{M}\times\mathbf{B}(t),

is a Hamiltonian system on the sphere :math:`S^2_{M_0}`. Standard explicit
Runge-Kutta integrators violate the conserved Casimir
:math:`\|\mathbf{M}\|^2=M_0^2` over long pulse trains, producing systematic
drift in cycle-Bloch / bottleneck training [Loktyushin 2021, Solomon 2023].

Munthe-Kaas Lie-group integrators (Munthe-Kaas 1999, Appl. Numer. Math.)
preserve this Casimir *exactly* by updating the magnetisation as

.. math::

    \mathbf{M}_{n+1} = \exp(\Delta t\,\Omega_n)\,\mathbf{M}_n,
    \qquad \Omega_n\in\mathfrak{so}(3),

where :math:`\exp:\mathfrak{so}(3)\to\mathrm{SO}(3)` is the closed-form
Rodrigues exponential. Combined with Strang splitting for the relaxation
sub-step, the overall integrator is second-order accurate in :math:`\Delta t`
and exactly energy-preserving in the no-relaxation limit.

Exports:

* :func:`hat_so3` / :func:`vee_so3` — Lie-algebra map :math:`\mathbf{v}\leftrightarrow[\mathbf{v}]_\times`.
* :func:`rodrigues_exp` — closed-form :math:`\exp(\Delta t\,\hat\Omega)\in\mathrm{SO}(3)`.
* :func:`munthe_kaas_step` — single time step of the precession sub-flow.
* :func:`strang_split_step` — precession + relaxation Strang composition.
* :func:`energy_drift` — diagnostic :math:`|\|\mathbf{M}_N\|-M_0|` over a pulse train.
"""

from __future__ import annotations

import torch

__all__ = [
    "energy_drift",
    "hat_so3",
    "munthe_kaas_step",
    "rodrigues_exp",
    "strang_split_step",
    "vee_so3",
]


def hat_so3(v: torch.Tensor) -> torch.Tensor:
    r"""Skew-symmetrise a 3-vector: :math:`\mathbf{v}\mapsto[\mathbf{v}]_\times\in\mathfrak{so}(3)`."""
    if v.shape[-1] != 3:
        raise ValueError(f"hat_so3 expects last-dim 3; got {v.shape[-1]}.")
    z = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
        ],
        dim=-2,
    )


def vee_so3(omega: torch.Tensor) -> torch.Tensor:
    r"""Inverse of :func:`hat_so3`: :math:`[\mathbf{v}]_\times\mapsto\mathbf{v}`."""
    if omega.shape[-2:] != (3, 3):
        raise ValueError(f"vee_so3 expects (..., 3, 3); got {tuple(omega.shape)}.")
    return torch.stack([omega[..., 2, 1], omega[..., 0, 2], omega[..., 1, 0]], dim=-1)


def rodrigues_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    r"""Closed-form Rodrigues formula
    :math:`\exp([\mathbf{v}]_\times)=I+\sin\theta\,[\hat{\mathbf{v}}]_\times+(1-\cos\theta)\,[\hat{\mathbf{v}}]_\times^2`.

    Args:
        axis_angle: ``(..., 3)`` Lie-algebra element :math:`\mathbf{v}=\theta\,\hat{\mathbf{v}}`.

    Returns:
        ``(..., 3, 3)`` rotation matrices.
    """
    if axis_angle.shape[-1] != 3:
        raise ValueError(f"axis_angle last dim must be 3; got {axis_angle.shape[-1]}.")
    theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = axis_angle / theta
    k_hat = hat_so3(axis)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand(
        *axis_angle.shape[:-1], 3, 3
    )
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    return eye + sin_theta * k_hat + (1.0 - cos_theta) * (k_hat @ k_hat)


def munthe_kaas_step(
    magnetisation: torch.Tensor,
    b_field: torch.Tensor,
    dt: float,
    gamma: float = 2 * torch.pi * 42.577e6,
) -> torch.Tensor:
    r"""One Munthe-Kaas step of the precession sub-flow.

    Update: :math:`\mathbf{M}_{n+1}=\exp(-\gamma\,\Delta t\,[\mathbf{B}]_\times)\,\mathbf{M}_n`,
    the exact flow of the module's law :math:`\dot{\mathbf{M}}=\gamma\,\mathbf{M}\times\mathbf{B}`
    (since :math:`\mathbf{M}\times\mathbf{B}=-[\mathbf{B}]_\times\mathbf{M}`).
    Preserves :math:`\|\mathbf{M}\|^2` exactly to machine precision (the
    exponential of a skew-symmetric matrix is orthogonal).

    Args:
        magnetisation: ``(..., 3)`` magnetisation vector.
        b_field: ``(..., 3)`` magnetic field at time :math:`t`.
        dt: Time step.
        gamma: Gyromagnetic ratio (default proton).

    Returns:
        ``(..., 3)`` updated magnetisation.
    """
    if magnetisation.shape != b_field.shape:
        raise ValueError("magnetisation and b_field must have identical shape.")
    if magnetisation.shape[-1] != 3:
        raise ValueError("last dim must be 3.")
    # dM/dt = γ M×B = −γ [B]_× M, so the flow is exp(−γΔt[B]_×). Using +γ here
    # would integrate dM/dt = +γ B×M (the negation of the module's stated law),
    # rotating M the wrong way about B (e.g. x̂→+ŷ for B=+ẑ instead of Larmor x̂→−ŷ).
    rotation = rodrigues_exp(-gamma * dt * b_field)
    return torch.einsum("...ij,...j->...i", rotation, magnetisation)


def strang_split_step(
    magnetisation: torch.Tensor,
    b_field: torch.Tensor,
    dt: float,
    t1: float,
    t2: float,
    m0_equilibrium: torch.Tensor | None = None,
    gamma: float = 2 * torch.pi * 42.577e6,
) -> torch.Tensor:
    r"""Strang-splitting step :math:`\Phi_{\Delta t}=\Phi^{\rm rot}_{\Delta t/2}\circ\Phi^{\rm relax}_{\Delta t}\circ\Phi^{\rm rot}_{\Delta t/2}`.

    Args:
        magnetisation: ``(..., 3)`` magnetisation at time :math:`t`.
        b_field: ``(..., 3)`` field at time :math:`t+\Delta t/2`.
        dt: Time step.
        t1: Longitudinal relaxation time :math:`T_1`.
        t2: Transverse relaxation time :math:`T_2`.
        m0_equilibrium: ``(..., 3)`` equilibrium magnetisation; defaults to
            :math:`(0,0,M_0)` with :math:`M_0` taken from the initial norm.
        gamma: Gyromagnetic ratio.

    Returns:
        ``(..., 3)`` magnetisation at time :math:`t+\Delta t`.
    """
    half = munthe_kaas_step(magnetisation, b_field, dt / 2, gamma=gamma)
    if m0_equilibrium is None:
        m0_norm = magnetisation.norm(dim=-1, keepdim=True)
        zeros = torch.zeros_like(magnetisation[..., :2])
        m0_equilibrium = torch.cat([zeros, m0_norm], dim=-1)
    decay_xy = torch.exp(torch.tensor(-dt / max(t2, 1e-12), dtype=magnetisation.dtype))
    decay_z = torch.exp(torch.tensor(-dt / max(t1, 1e-12), dtype=magnetisation.dtype))
    relaxed_xy = decay_xy * half[..., :2]
    relaxed_z = decay_z * half[..., 2] + (1.0 - decay_z) * m0_equilibrium[..., 2]
    relaxed = torch.cat([relaxed_xy, relaxed_z.unsqueeze(-1)], dim=-1)
    return munthe_kaas_step(relaxed, b_field, dt / 2, gamma=gamma)


def energy_drift(magnetisation_trajectory: torch.Tensor, m0: float) -> torch.Tensor:
    r"""Diagnostic :math:`\max_t|\|\mathbf{M}(t)\|-M_0|`.

    Args:
        magnetisation_trajectory: ``(T, ..., 3)`` trajectory across time.
        m0: Equilibrium magnetisation norm.

    Returns:
        Scalar maximum-over-time energy drift.
    """
    norms = magnetisation_trajectory.norm(dim=-1)
    drift = (norms - m0).abs().max()
    return drift
