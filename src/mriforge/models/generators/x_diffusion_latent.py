"""X-Diffusion Latent Generator for cross-distribution bridge models.

Latent-space diffusion bridge connecting two modality distributions (e.g. T1 → T2).
Uses a UNet backbone conditioned on sinusoidal timestep embeddings to predict
the velocity field / noise for a flow-matching or DDPM reverse process.

Reference:
    Liu et al., "Flow Matching for Generative Modeling", ICLR 2023.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _ResBlock(nn.Module):
    """Residual block with optional time-embedding injection."""

    def __init__(self, ch: int, time_dim: int | None = None) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, ch) if time_dim else None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward method for _ResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(t_emb)[..., None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


@register_model(name="x_diffusion_latent", training_mode="diffusion")
class XDiffusionLatentGenerator(IGenerator, nn.Module):
    """Cross-Diffusion Latent Bridge Generator.

    Operates in latent space to bridge two distinct modality distributions
    using flow matching or DDPM-style reverse process.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        features: list[int] = kwargs.get("features", [32, 64, 128, 256])
        time_dim = features[-1] * 4

        # Sinusoidal timestep MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self._time_dim = time_dim

        # Encoder
        self.intro = nn.Conv2d(in_channels, features[0], 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch_prev = features[0]
        for ch in features[1:]:
            self.encoders.append(_ResBlock(ch_prev, time_dim))
            self.downs.append(nn.Conv2d(ch_prev, ch, 4, stride=2, padding=1))
            ch_prev = ch

        # Middle
        self.mid = _ResBlock(ch_prev, time_dim)

        # Decoder
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ch in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch_prev, ch, 4, stride=2, padding=1))
            self.decoders.append(_ResBlock(ch * 2, time_dim))  # concat skip
            ch_prev = ch * 2  # after concat

        self.out_conv = nn.Conv2d(ch_prev, out_channels, 3, padding=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "x_diffusion_latent"

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Predict velocity field / noise.

        forward method for XDiffusionLatentGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        t_emb = None
        if timesteps is not None:
            t_emb = self._sinusoidal_embedding(timesteps, self._time_dim)
            t_emb = self.time_mlp(t_emb)

        h = self.intro(x)
        skips: list[torch.Tensor] = []
        for enc, down in zip(self.encoders, self.downs, strict=False):
            h = enc(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid(h, t_emb)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=False):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            h = dec(h, t_emb)

        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def encode_to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Identity encoder — this generator operates in pixel space.

        F-XDIFF-LATENT-ENCODE / 2026-05-20 — the model_type
        ``x_diffusion_latent`` triggers DiffusionTrainingStrategy's
        ``_is_latent_diffusion`` heuristic (substring "latent" +
        "diffusion") which then expects ``encode_to_latent`` on the
        generator. The architecture, however, is a plain pixel-space
        U-Net (no VAE encoder/decoder pair) — the "Latent" in the
        class name refers to the *cross-modality bridge* objective,
        not to operating in a compressed latent. Smoke 20260519
        crashed at iter-1 with the explicit fix-hint.

        Returning the input unchanged makes the latent-diffusion
        wrapper a no-op for this model. The downstream loss compares
        ``pred`` to ``target`` in pixel space, which is the intended
        behaviour.
        """
        return x

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Identity decoder — pair of :meth:`encode_to_latent`."""
        return z

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
