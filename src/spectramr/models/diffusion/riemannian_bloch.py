"""Riemannian Bloch diffusion model — Idea 5.

A coordinate-style network that takes a noised manifold point
``θ_t ∈ M`` (3-vector: M0, T1, T2), the diffusion time ``t``, and an
observation context, and predicts the manifold-tangent score
``u_φ(θ_t, t) ∈ T_{θ_t} M``. The output is projected onto the tangent
space via the manifold's ``project_tangent`` method.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold
from spectramr.models.registry import register_model


class _SinusoidalEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of a scalar."""

    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim
        freqs = torch.exp(
            -torch.arange(0, dim, 2).float() * (torch.log(torch.tensor(10000.0)) / dim)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            x = x.unsqueeze(0)
        emb = x.unsqueeze(-1) * self.freqs
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


@register_model(
    "riemannian_diffusion_qmap",
    training_mode="diffusion",
    spatial_dims=(2,),
)
class RiemannianDiffusionQMap(nn.Module):
    """Tangent-space score predictor on the Bloch relaxation manifold.

    Args:
        cond_channels: Number of conditioning context channels
            (e.g. multi-TE T2-weighted volume packed into channels).
        time_dim: Diffusion time embedding dimension.
        hidden: Hidden layer width.
        n_layers: Number of MLP layers.
    """

    def __init__(
        self,
        cond_channels: int = 0,
        time_dim: int = 32,
        hidden: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.time_embed = _SinusoidalEmbedding(time_dim)
        in_dim = 3 + time_dim + cond_channels
        layers: list[nn.Module] = []
        d_in = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d_in, hidden))
            layers.append(nn.GELU())
            d_in = hidden
        layers.append(nn.Linear(hidden, 3))
        self.net = nn.Sequential(*layers)
        self.manifold = BlochRelaxationManifold()

    def forward(
        self,
        theta_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Return tangent-space score at ``theta_t``.

        Args:
            theta_t: [B, 3] points on M.
            t: [B] diffusion times.
            cond: Optional [B, cond_channels] conditioning vector.

        Returns:
            [B, 3] tangent vectors at ``theta_t``.
        """
        if t.dim() == 0:
            t = t.expand(theta_t.shape[0])
        t_emb = self.time_embed(t)
        if cond is None:
            inputs = torch.cat([theta_t, t_emb], dim=-1)
        else:
            inputs = torch.cat([theta_t, t_emb, cond], dim=-1)
        raw = self.net(inputs)
        return self.manifold.project_tangent(theta_t, raw)


__all__ = ["RiemannianDiffusionQMap"]
