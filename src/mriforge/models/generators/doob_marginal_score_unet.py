"""Unconditional 7T-marginal score network for the B-1.10 score-based SDEdit bridge.

An UNCONDITIONAL time-conditioned eps-net trained by denoising score matching on the 7T
target marginal ALONE (no source, no field conditioning). Its prediction gives the marginal
score ``grad log p_t^{7T}(x) = -eps_theta(x,t) / sqrt(1-alphabar_t)`` — the spec's
``grad log h`` (B-1.10's "condition to hit the 7T density h"). For a translation task the
terminal is unknown, so the realisable pin is to the 7T MARGINAL via this score; adding it to
the reverse drift is Anderson's reverse SDE (score-based reverse sampling), so the bridge is
operationally marginal-score-guided SDEdit, not a separately-ablatable terminal correction
(see :mod:`mriforge.infrastructure.training.strategies.doob_bridge_strategy`). The source
enters ONLY as the sampler's initial condition (a partially-noised source), never as a
network input — that keeps this distinct from the source+field-conditioned CFG net of B-3.5
(:class:`FieldGuidedScoreUNet`) and the velocity-matching bridge of B-3.3.

Magnitude-only (1->1); raises on complex. ``timesteps`` is a REQUIRED keyword arg (no
default) so the Tier-2 synthetic forward probe injects it (a defaulted arg would be skipped).
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


@register_model(name="doob_marginal_score_unet", training_mode="doob_bridge")
class DoobMarginalScoreUNet(nn.Module):
    """Time-conditioned UNCONDITIONAL eps-net = the 7T marginal score (Doob ``h``)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        time_dim: int = 64,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "DoobMarginalScoreUNet is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )
        self.body_in = nn.Conv2d(in_channels, self.width, 3, padding=1)
        self.blocks = nn.ModuleList(
            _conv_block(self.width, self.width) for _ in range(self.n_blocks)
        )
        # Per-block time-FiLM heads (gamma, beta) from the time embedding; zero-init so the
        # block starts as identity-modulated (gamma=1, beta=0) -> stable training start.
        self.film = nn.ModuleList(nn.Linear(time_dim, 2 * self.width) for _ in range(self.n_blocks))
        for lin in self.film:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.body_out = nn.Conv2d(self.width, out_channels, 1)

    def forward(self, x: torch.Tensor, *, timesteps: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("DoobMarginalScoreUNet expects magnitude (real) input.")
        t_emb = self.time_embed(timesteps.float().reshape(-1))  # [B, time_dim]
        h = self.body_in(x)
        for block, film in zip(self.blocks, self.film, strict=True):
            r = block(h)
            gamma, beta = film(t_emb).chunk(2, dim=-1)  # [B, width] each
            r = r * (1.0 + gamma)[:, :, None, None] + beta[:, :, None, None]
            h = h + r  # residual
        return self.body_out(h)
