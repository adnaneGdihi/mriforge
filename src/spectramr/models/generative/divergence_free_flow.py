"""Divergence-free flow-matching velocity field.

A flow-matching model whose predicted velocity ``v_theta(x_t, t)`` is
structurally divergence-free: rather than penalising ``div v`` with a soft
loss, the network predicts a streamfunction (2D) or vector potential (3D)
and the velocity is obtained as its curl, which is divergence-free by
construction (up to finite-difference discretisation error). This is
strictly stronger than the existing ``divergence_free`` *loss*, which only
penalises divergence.

Reference: incompressible flow-matching; streamfunction/curl construction
(Harlow & Welch MAC grid, 1965); flow matching (Lipman et al., 2023).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.blocks.streamfunction import (
    CurlOf3DVectorPotential,
    StreamfunctionTo2DVelocity,
)
from spectramr.models.registry import register_model


class _PsiBackbone(nn.Module):
    """Small time-conditioned CNN producing the (vector) potential ``psi``.

    Outputs 1 channel in 2D (scalar streamfunction) or 3 channels in 3D
    (vector potential). Time is injected via a sinusoidal-free scalar
    broadcast added as an extra input channel.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden: int, ndim: int) -> None:
        super().__init__()
        conv = nn.Conv2d if ndim == 2 else nn.Conv3d
        # +1 channel for the broadcast time scalar.
        self.net = nn.Sequential(
            conv(in_channels + 1, hidden, 3, padding=1),
            nn.SiLU(),
            conv(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            conv(hidden, out_channels, 3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Broadcast t (shape [B] or [B,1]) into a constant feature map.
        b = x_t.shape[0]
        t_flat = t.reshape(b, *([1] * (x_t.dim() - 1)))
        t_map = t_flat.expand(b, 1, *x_t.shape[2:])
        return self.net(torch.cat([x_t, t_map], dim=1))


@register_model(
    name="divergence_free_flow",
    training_mode="flow_matching",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class DivergenceFreeFlow(nn.Module):
    """Structurally divergence-free flow-matching velocity field.

    Args:
        in_channels: Channels of the state ``x_t`` (e.g. 2 for a 2D
            velocity, 3 for a 3D velocity). Defaults to 2.
        out_channels: Channels of the predicted velocity. Must equal the
            spatial dimension (``2`` in 2D, ``3`` in 3D).
        ndim: Spatial rank, ``2`` or ``3``.
        hidden_channels: Width of the potential backbone.

    Raises:
        ValueError: For unsupported ``ndim`` or an ``out_channels`` that
            does not match the velocity dimension (no silent fallback).
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        ndim: int = 2,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        if ndim not in (2, 3):
            raise ValueError(f"ndim must be 2 or 3, got {ndim}.")
        if out_channels != ndim:
            raise ValueError(
                f"Velocity has {ndim} components in {ndim}D; "
                f"out_channels must equal {ndim}, got {out_channels}."
            )
        self.ndim = int(ndim)
        # psi is scalar in 2D, a 3-vector in 3D.
        psi_channels = 1 if ndim == 2 else 3
        self.psi_net = _PsiBackbone(int(in_channels), psi_channels, int(hidden_channels), ndim)
        self.to_velocity: nn.Module = (
            StreamfunctionTo2DVelocity() if ndim == 2 else CurlOf3DVectorPotential()
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict a divergence-free velocity ``v_theta(x_t, t)``.

        Args:
            x_t: State tensor, ``[B, C, H, W]`` (2D) or ``[B, C, D, H, W]`` (3D).
            t: Per-sample time, broadcastable to ``[B]``.
        """
        psi = self.psi_net(x_t, t)
        return self.to_velocity(psi)
