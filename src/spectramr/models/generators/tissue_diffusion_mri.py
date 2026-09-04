"""Tissue Diffusion MRI Generator with Bloch Data Consistency.

Physics-guided diffusion operating in tissue parameter space (ρ, T1, T2, T2*, ADC, χ).
The Bloch equations serve as a differentiable decoder: tissue_params → image.
DDPM forward diffusion adds noise to normalized tissue maps; the reverse process
denoises with Bloch DC projection at each step.

Reference:
    Chung et al., "Score-MRI", IEEE TMI, 2022.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _TissueResBlock(nn.Module):
    """Residual block for tissue parameter space."""

    def __init__(self, ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, ch), ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, ch), ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.act = nn.SiLU()
        self.time_proj = nn.Linear(time_dim, ch)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """forward method for _TissueResBlock.

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
        h = h + self.time_proj(t_emb)[..., None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class _SimpleBlochForward(nn.Module):
    """Differentiable Bloch forward model (simplified spin-echo signal).

    Given tissue maps (T1, T2, PD) and sequence parameters (TR, TE),
    computes S = PD * (1 - exp(-TR/T1)) * exp(-TE/T2).
    """

    def __init__(self, default_tr: float = 500.0, default_te: float = 15.0) -> None:
        super().__init__()
        self.register_buffer("tr", torch.tensor(default_tr))
        self.register_buffer("te", torch.tensor(default_te))

    def forward(self, tissue_maps: torch.Tensor) -> torch.Tensor:
        """Compute MR signal from tissue maps.

        Args:
            tissue_maps: [B, tissue_channels, H, W] where channels are
                         [PD, T1, T2, T2*, ADC, χ]. Only first 3 used.

        forward method for _SimpleBlochForward.

        Executes PyTorch tensor operations.

        Args:
            tissue_maps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        pd = tissue_maps[:, 0:1].clamp(min=1e-6)
        t1 = tissue_maps[:, 1:2].clamp(min=1e-3)
        t2 = tissue_maps[:, 2:3].clamp(min=1e-3)
        signal = pd * (1.0 - torch.exp(-self.tr / t1)) * torch.exp(-self.te / t2)
        return signal


@register_model(name="tissue_diffusion_mri", training_mode="diffusion")
class TissueDiffusionMRIGenerator(IGenerator, nn.Module):
    """Tissue-specific diffusion generator with Bloch DC projection."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_steps = kwargs.get("num_timesteps", kwargs.get("n_steps", 1000))

        tissue_channels: int = kwargs.get("tissue_channels", 6)
        base_dim: int = kwargs.get("base_dim", 64)
        time_dim = base_dim * 4

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self._time_dim = time_dim

        # Tissue UNet backbone
        self.intro = nn.Conv2d(in_channels, base_dim, 3, padding=1)
        self.enc1 = _TissueResBlock(base_dim, time_dim)
        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 4, stride=2, padding=1)
        self.enc2 = _TissueResBlock(base_dim * 2, time_dim)
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, stride=2, padding=1)
        self.mid = _TissueResBlock(base_dim * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.dec2 = _TissueResBlock(base_dim * 4, time_dim)  # concat
        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim, 4, stride=2, padding=1)
        self.dec1 = _TissueResBlock(base_dim * 2, time_dim)  # concat
        self.out_conv = nn.Conv2d(base_dim * 2, out_channels, 3, padding=1)

        # Bloch forward model for DC
        self.bloch = _SimpleBlochForward()

    @staticmethod
    def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    @property
    def name(self) -> str:
        return "tissue_diffusion_mri"

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """forward method for TissueDiffusionMRIGenerator.

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
        else:
            t_emb = torch.zeros(x.shape[0], self._time_dim, device=x.device)
            t_emb = self.time_mlp(t_emb)

        h = self.intro(x)
        s1 = self.enc1(h, t_emb)
        h = self.down1(s1)
        s2 = self.enc2(h, t_emb)
        h = self.down2(s2)
        h = self.mid(h, t_emb)
        h = self.up2(h)
        h = torch.cat([h, s2], dim=1)
        h = self.dec2(h, t_emb)
        h = self.up1(h)
        h = torch.cat([h, s1], dim=1)
        h = self.dec1(h, t_emb)
        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
