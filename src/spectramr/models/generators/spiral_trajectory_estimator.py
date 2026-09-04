"""Spiral trajectory-recovery estimator (exp_vf_35 G1).

Estimates the spiral k-space trajectory deviation ``Δk(t)`` induced by gradient
delays / eddy currents / the gradient impulse response (GIRF). The GIRF is a
*scanner* property shared across acquisitions, so θ is a learnable parameter set
(the standard calibration), not a per-scan hypernetwork.

Forward model: ``k_actual = gamma ∫ g_actual``, ``g_actual = h * g_nominal``. The
estimator reconstructs ``Δk̂``:

    g_nom = d/dt k_nominal  →  g_act = GIRF(g_nom)  →  k_act = ∫ g_act
    Δk̂   = k_act - k_nom   (+ a constant k0 offset)

At the identity GIRF (delta kernel, ``k0 = 0``) ``g_act = g_nom`` so ``Δk̂ = 0``
— the θ=θ0 / Δk=0 consistency limit (the baseline). Every op is pure-torch
(conv1d + cumsum), so a SUPERVISED ``‖Δk̂ - Δk_true‖`` loss trains θ — the only
differentiable path, since torchkbnufft does not pass gradients to the
trajectory coordinates (a through-NUFFT objective cannot train θ; wiring one
would be an inert mechanism, pitfall #16). The estimate is exposed as
``last_trajectory_estimate`` in the SPIRAL parametrisation ``[B, N, 2]`` — never
a Cartesian per-readout-line ``Δk`` (the diff_trajectory_opt mismatch).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn.functional import pad

from spectramr.infrastructure.physics.girf import GIRFKernel
from spectramr.models.registry import register_model

__all__ = ["SpiralTrajectoryEstimator"]


@register_model(
    name="spiral_trajectory_estimator",
    training_mode="trajectory_recon",
    spatial_dims=(2,),
    # input/output_domain intentionally unannotated: this model consumes the
    # strategy-threaded *nominal trajectory* (a [B,N,2] coordinate tensor), not
    # the dataset's primary domain, so the static data→model domain check does
    # not apply. trajectory_parametrization IS declared so the metric↔param lock
    # (pitfall #16) binds.
    accepts_complex=False,
    requires_paired_data=False,
    trajectory_parametrization="spiral",
)
class SpiralTrajectoryEstimator(nn.Module):
    """GIRF/delay estimator: nominal spiral trajectory → deviation ``Δk̂(t)``."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        *,
        girf_param_dim: int = 65,
        n_axes: int = 2,
        freeze_estimator: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_axes = n_axes
        self.freeze_estimator = freeze_estimator
        # Learnable GIRF FIR per axis (init = delta → identity → Δk̂=0 at θ0).
        self.girf = GIRFKernel(n_axes=n_axes, kernel_size=girf_param_dim)
        # Constant per-axis k-space offset (the k0 / B0-as-k0 term).
        self.k0 = nn.Parameter(torch.zeros(n_axes))
        if freeze_estimator:
            # Δk=0 / no-estimate baseline (pitfall #17): freeze θ at the identity
            # so Δk̂≡0 (the nominal-trajectory control the trainable arm must beat).
            for p in self.parameters():
                p.requires_grad_(False)
        self.last_trajectory_estimate: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return "spiral_trajectory_estimator"

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, k_nominal: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        knom = k_nominal.float()
        squeeze_batch = knom.dim() == 2  # [N, 2] single arm
        if squeeze_batch:
            knom = knom.unsqueeze(0)  # → [1, N, 2]
        if knom.shape[-1] != self.n_axes:
            raise ValueError(
                f"spiral_trajectory_estimator expects coords last [..,N,{self.n_axes}], "
                f"got trailing dim {knom.shape[-1]}."
            )
        # [B, N, A] → [B, A, N] for conv1d over the readout
        kt = knom.transpose(1, 2)
        # nominal gradient = finite difference along the readout (front-padded)
        g_nom = pad(kt[..., 1:] - kt[..., :-1], (1, 0))  # [B, A, N]
        g_act = self.girf(g_nom)  # GIRF-corrected gradient (identity at init)
        k_act = torch.cumsum(g_act, dim=-1)
        k_nom_int = torch.cumsum(g_nom, dim=-1)  # = k_nom - k_nom[0] (telescoping)
        dk = (k_act - k_nom_int) + self.k0.view(1, -1, 1)  # [B, A, N] deviation
        dk = dk.transpose(1, 2)  # → [B, N, A] spiral parametrisation
        if squeeze_batch:
            dk = dk.squeeze(0)
        self.last_trajectory_estimate = dk.detach()
        return dk
