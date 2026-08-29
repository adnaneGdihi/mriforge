r"""Blurring diffusion model (Hoogeboom & Salimans, ICLR 2023).

Heat-equation forward process that destroys high-frequency content via
the DCT-diagonalised heat equation rather than additive Gaussian noise,
trained with a v-parameterised denoiser.

The model bundles:
    * a time-conditioned U-Net denoiser (``v_\theta(x_t, t)``), built
      internally from the shared time-aware blocks in
      :mod:`mriforge.models.diffusion.noise_schedules`, and
    * a :class:`~mriforge.models.diffusion.schedules.blurring_schedule.BlurringSchedule`
      constructed from scalar kwargs.

It is registered as ``blurring_diffusion`` (training mode ``diffusion``)
and is constructible with no arguments, so the model factory's
kwarg-filtered ``create_generator`` path works out of the box.

References:
    E. Hoogeboom and T. Salimans, "Blurring diffusion models," ICLR 2023.
    T. Salimans and J. Ho, "Progressive distillation for fast sampling of
    diffusion models," ICLR 2022 (v-parameterisation).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.diffusion.schedules.blurring_schedule import BlurringSchedule
from mriforge.models.registry import register_model

__all__ = [
    "BlurringDiffusion",
    "BlurringUNet",
    "DoubleConvWithTime",
    "DownWithTime",
    "SinusoidalTimeEmbedding",
    "UpWithTime",
]


class DoubleConvWithTime(nn.Module):
    """Two 3x3 conv-GN-SiLU blocks with an additive timestep embedding.

    The time embedding is projected to ``out_channels`` and added after the
    first convolution (the standard DDPM residual-block time injection).
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.norm2(self.conv2(h)))
        return h


class DownWithTime(nn.Module):
    """2x downsample (max-pool) followed by a time-conditioned double conv."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithTime(in_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x), t_emb)


class UpWithTime(nn.Module):
    """2x upsample (``in→in//2``) + skip concat + time-conditioned double conv.

    The skip connection carries ``in_channels // 2`` channels (symmetric
    U-Net), so the post-concat channel count is ``in_channels`` and the
    double conv maps ``in_channels → out_channels``.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvWithTime(in_channels, out_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if odd spatial dims caused a 1-pixel mismatch with the skip.
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([skip, x], dim=1), t_emb)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal timestep embedding."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embedding dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map integer timesteps ``[B]`` to embeddings ``[B, dim]``."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class BlurringUNet(nn.Module):
    """Minimal time-conditioned U-Net denoiser for blurring diffusion.

    Reuses the shared ``*WithTime`` blocks so the time-embedding plumbing
    matches the rest of the diffusion stack. Predicts the v-target with
    the same channel count as the input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        c = base_channels
        self.inc = DoubleConvWithTime(in_channels, c, time_emb_dim=time_emb_dim)
        self.down1 = DownWithTime(c, c * 2, time_emb_dim=time_emb_dim)
        self.down2 = DownWithTime(c * 2, c * 4, time_emb_dim=time_emb_dim)
        self.bot = DoubleConvWithTime(c * 4, c * 4, time_emb_dim=time_emb_dim)
        self.up1 = UpWithTime(c * 4, c * 2, time_emb_dim=time_emb_dim)
        self.up2 = UpWithTime(c * 2, c, time_emb_dim=time_emb_dim)
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the v-target given a blurred image and timestep."""
        t_emb = self.time_mlp(t)
        x1 = self.inc(x, t_emb)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        xb = self.bot(x3, t_emb)
        x = self.up1(xb, x2, t_emb)
        x = self.up2(x, x1, t_emb)
        return self.outc(x)


@register_model(
    name="blurring_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class BlurringDiffusion(nn.Module):
    r"""Heat-equation blurring diffusion with v-parameterised denoiser.

    Constructible with no arguments: a :class:`BlurringUNet` denoiser and
    a :class:`BlurringSchedule` are built from the scalar kwargs.

    Args:
        in_channels: Input image channels.
        out_channels: Output (v-prediction) channels.
        num_timesteps: Number of diffusion steps :math:`T`.
        image_size: Spatial side length :math:`L` used to build the
            DCT eigenvalue grid.
        sigma_min: Residual Gaussian std at :math:`t = 0`.
        sigma_max: Residual Gaussian std at the final step.
        base_channels: U-Net width.
        time_embedding_dim: Sinusoidal time-embedding dimension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_timesteps: int = 1000,
        image_size: int = 256,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1,
        base_channels: int = 64,
        time_embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps

        self.unet = BlurringUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_emb_dim=time_embedding_dim,
        )
        self.schedule = BlurringSchedule(
            num_timesteps=num_timesteps,
            image_size=image_size,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the v-target for a blurred image ``x_t`` at step ``t``.

        Args:
            x_t: Blurred + noised image ``[B, C, H, W]``.
            t: Long timestep indices ``[B]``.

        Returns:
            Predicted v-target of the same spatial shape as ``x_t``.
        """
        return self.unet(x_t, t)

    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward (blur + residual-noise) process; see ``BlurringSchedule``."""
        if noise is None:
            noise = torch.randn_like(x0)
        return self.schedule.q_sample(x0, t, noise)

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        r"""v-parameterised MSE training objective.

        Draws :math:`t \sim \mathrm{Uniform}\{0, \dots, T-1\}`, blurs
        ``x0`` to ``x_t`` via the heat-equation forward process, and
        regresses the denoiser onto the v-target.

        Args:
            x0: Clean image batch ``[B, C, H, W]``.

        Returns:
            Scalar MSE loss between predicted and true v-targets.
        """
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        v_target = self.schedule.v_target(x0, noise, t)
        v_pred = self.unet(x_t, t)
        return F.mse_loss(v_pred, v_target)
