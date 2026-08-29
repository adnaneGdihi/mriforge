"""Source-conditioned residual score network for the B-1.10 residual detail bridge.

A time-conditioned eps-net that predicts the score of the source->7T RESIDUAL
``r = x_7T - x_src`` CONDITIONED on the source image. Where
:class:`~mriforge.models.generators.doob_marginal_score_unet.DoobMarginalScoreUNet`
learns the UNCONDITIONAL 7T marginal (so its high-frequency detail is subject-generic),
this net sees the source as an extra input channel, so the residual it denoises is tied to
THIS subject's anatomy. Because the mrixfields source/target volumes are paired, registered,
and normalized on the shared grid, the residual is small and mostly high-frequency: learning
it concentrates the net's capacity on the detail the bare bridge misses, and the sampler
composes ``out = source + r`` so the coarse anatomy is the exact registered source.

Distinct from :class:`FieldGuidedScoreUNet` (B-3.5): that net conditions on source+field and
generates the FULL target with classifier-free field guidance; this net has no field/CFG and
predicts the RESIDUAL. Magnitude-only (1->1 for ``x``; ``cond_image`` is a 1-channel source
condition); raises on complex. ``timesteps`` is a REQUIRED keyword arg (no default) so the
Tier-2 synthetic forward probe injects it (a defaulted arg would be skipped).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.embeddings import SinusoidalPositionEmbedding
from mriforge.models.registry import register_model


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(name="doob_residual_score_unet", training_mode="doob_bridge")
class DoobResidualScoreUNet(nn.Module):
    """Time-conditioned eps-net for the source-conditioned 7T RESIDUAL score."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        cond_channels: int = 1,
        width: int = 64,
        time_dim: int = 64,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "DoobResidualScoreUNet is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if cond_channels != 1:
            raise ValueError(
                f"DoobResidualScoreUNet takes one source condition channel; got {cond_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        self.width = int(width)
        self.cond_channels = int(cond_channels)
        self.n_blocks = int(n_blocks)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        # Body input takes [x_t (noised residual), source] concatenated on the channel axis.
        self.body_in = nn.Conv2d(in_channels + self.cond_channels, self.width, 3, padding=1)
        self.blocks = nn.ModuleList(
            _conv_block(self.width, self.width) for _ in range(self.n_blocks)
        )
        # Per-block time-FiLM heads (gamma, beta); zero-init -> identity-modulated start.
        self.film = nn.ModuleList(nn.Linear(time_dim, 2 * self.width) for _ in range(self.n_blocks))
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.body_out = nn.Conv2d(self.width, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        cond_image: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x) or torch.is_complex(cond_image):
            raise ValueError("DoobResidualScoreUNet expects magnitude (real) input.")
        if cond_image.shape[-2:] != x.shape[-2:] or cond_image.shape[0] != x.shape[0]:
            raise ValueError(
                "cond_image (source) must match x in batch and spatial dims; "
                f"got x={tuple(x.shape)}, cond_image={tuple(cond_image.shape)}."
            )
        t_emb = self.time_embed(timesteps.float().reshape(-1))  # [B, time_dim]
        h = self.body_in(torch.cat([x, cond_image], dim=1))
        for block, film in zip(self.blocks, self.film, strict=True):
            r = block(h)
            gamma, beta = film(t_emb).chunk(2, dim=-1)  # [B, width] each
            r = r * (1.0 + gamma)[:, :, None, None] + beta[:, :, None, None]
            h = h + r  # residual
        return self.body_out(h)
