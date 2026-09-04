"""Teichmüller-geodesic schedule head for k-space cold diffusion (§6).

Tiny MLP ``r_ψ : [0,1] → [0,1]`` that parameterises the radial
position along the closed-form Teichmüller geodesic between two
Beltrami coefficients :math:`\\mu_0, \\mu_T` (the SFCs at the two
diffusion endpoints). Combined with
:func:`teichmuller_geodesic`, the head produces a step-dependent
Beltrami coefficient :math:`\\mu_t = \\gamma(\\mu_0, \\mu_T, r_\\psi(t))`
that stays inside the Beltrami unit ball.

Used by the existing ``cold_diffusion`` training strategy when
``training.kspace_cold.schedule_type: teichmuller_adaptive``.

References:
    [9] Gardiner & Lakic, *Quasiconformal Teichmüller Theory*, AMS, 2000.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.conformal_geometry import teichmuller_geodesic


class TeichmullerScheduleHead(nn.Module):
    """Two-layer MLP producing ``r(t) ∈ [0, 1]``.

    The output activation is ``sigmoid`` so the radial parameter is
    natively bounded; an additional scale factor ``r_max < 1``
    guarantees we never request the boundary value where the geodesic
    is degenerate.
    """

    def __init__(self, hidden: int = 16, r_max: float = 0.99) -> None:
        super().__init__()
        self.r_max = r_max
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map ``t[B]`` in ``[0, 1]`` to ``r(t)[B]`` in ``[0, r_max]``."""
        t_in = t.view(-1, 1).to(self.net[0].weight.dtype)
        return self.r_max * torch.sigmoid(self.net(t_in)).squeeze(-1)


def teichmuller_mu_schedule(mu0: torch.Tensor, muT: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Convenience: closed-form ``μ_t`` schedule given per-step ``r``.

    Broadcasts ``r[T]`` against ``mu0, muT[H, W]`` (or any matching
    spatial shape) to produce ``μ_t[T, H, W]``.
    """
    # r[T] → [T, 1, ..., 1] matching mu0's rank.
    r_b = r.to(mu0.dtype)
    while r_b.dim() < mu0.dim() + 1:
        r_b = r_b.unsqueeze(-1)
    # mu0, muT broadcast as [1, *spatial] across the leading T axis.
    return teichmuller_geodesic(mu0.unsqueeze(0), muT.unsqueeze(0), r_b)


__all__ = ["TeichmullerScheduleHead", "teichmuller_mu_schedule"]
