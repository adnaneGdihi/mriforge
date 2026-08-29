"""Score-field U-Net for joint (x, c) score estimation (Idea 2).

The architecture is a small U-Net with FiLM conditioning that produces
two output heads: ``s_x`` and ``s_c``. The first is the standard
image-domain score; the second is the contrast-direction score that
enables protocol inference at sampling time.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.registry import register_model


class _FiLMBlock(nn.Module):
    """Conv-GN-GELU with FiLM modulation from a conditioning embedding."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.film = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(self.conv(x))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return self.act(h * (1.0 + gamma) + beta)


@register_model(
    "score_field_unet",
    training_mode="diffusion",
    supports_contrast_conditioning=True,
    spatial_dims=(2,),
)
class ScoreFieldUNet(nn.Module):
    """Joint score U-Net with two heads ``s_x`` and ``s_c``.

    Args:
        in_channels: Image channels.
        cond_dim: Dim of the acquisition-parameter vector
            (TR, TE, TI, α, B0 ⇒ 5).
        base_width: Channel multiplier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        cond_dim: int = 5,
        base_width: int = 32,
    ) -> None:
        super().__init__()
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim + 1, 4 * base_width),
            nn.GELU(),
            nn.Linear(4 * base_width, 4 * base_width),
        )
        self.lift = nn.Conv2d(in_channels, base_width, 3, padding=1)
        self.block1 = _FiLMBlock(base_width, 4 * base_width)
        self.down = nn.Conv2d(base_width, 2 * base_width, 2, stride=2)
        self.block2 = _FiLMBlock(2 * base_width, 4 * base_width)
        self.up = nn.ConvTranspose2d(2 * base_width, base_width, 2, stride=2)
        self.block3 = _FiLMBlock(base_width, 4 * base_width)
        self.head_sx = nn.Conv2d(base_width, in_channels, 1)
        self.head_sc = nn.Linear(base_width, cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        c: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Image tensor [B, C, H, W].
            sigma: Diffusion noise level [B] or [B, 1].
            c: Acquisition parameters [B, cond_dim].

        Returns:
            ``(s_x, s_c)`` where ``s_x`` has shape [B, C, H, W] and
            ``s_c`` has shape [B, cond_dim].
        """
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        cond_in = torch.cat([c, sigma], dim=-1)
        cond = self.cond_embed(cond_in)
        h0 = self.lift(x)
        h1 = self.block1(h0, cond)
        d = self.down(h1)
        d = self.block2(d, cond)
        u = self.up(d)
        u = self.block3(u + h1, cond)
        s_x = self.head_sx(u)
        pooled = u.mean(dim=(-2, -1))
        s_c = self.head_sc(pooled)
        return s_x, s_c


__all__ = ["ScoreFieldUNet"]
