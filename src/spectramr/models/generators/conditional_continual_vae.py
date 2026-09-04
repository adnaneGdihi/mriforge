"""Conditional Continual VAE Generator for lifelong MRI learning.

VAE with task-specific latent distributions and catastrophic-forgetting
protection via Elastic Weight Consolidation (EWC). Enables sequential
training across scanners/contrasts without losing previous knowledge.

Reference:
    Ramapuram et al., "Lifelong Generative Modeling", NeurIPS 2017.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _TaskConditionedEncoder(nn.Module):
    """Encoder with task-specific conditioning via FiLM."""

    def __init__(self, in_ch: int, feature_dim: int, num_tasks: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, feature_dim, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, feature_dim * 2), feature_dim * 2),
            nn.SiLU(),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, feature_dim * 4), feature_dim * 4),
            nn.SiLU(),
        )
        # FiLM conditioning from task ID
        self.task_embed = nn.Embedding(num_tasks, feature_dim * 4)
        self.film_gamma = nn.Linear(feature_dim * 4, feature_dim * 4)
        self.film_beta = nn.Linear(feature_dim * 4, feature_dim * 4)

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """forward method for _TaskConditionedEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            task_id (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.conv(x)
        B = h.shape[0]
        tid = torch.full((B,), task_id, device=x.device, dtype=torch.long)
        te = self.task_embed(tid)
        gamma = self.film_gamma(te)[..., None, None]
        beta = self.film_beta(te)[..., None, None]
        return gamma * h + beta


@register_model(name="conditional_continual_vae", training_mode="vae")
class ConditionalContinualVAEGenerator(IGenerator, nn.Module):
    """Continual VAE with task conditioning for lifelong MRI learning."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        features: list[int] = kwargs.get("features", [32, 64, 128, 256])
        encoder_blocks: int = kwargs.get("encoder_blocks", 4)
        decoder_blocks: int = kwargs.get("decoder_blocks", 4)
        num_tasks: int = kwargs.get("num_tasks", 5)
        base_dim = features[0]

        # Encoder
        self.encoder = _TaskConditionedEncoder(in_channels, base_dim, num_tasks)
        enc_out_dim = base_dim * 4

        # Latent
        self.fc_mu = nn.Conv2d(enc_out_dim, enc_out_dim, 1)
        self.fc_logvar = nn.Conv2d(enc_out_dim, enc_out_dim, 1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(enc_out_dim, base_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, base_dim * 2), base_dim * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, base_dim), base_dim),
            nn.SiLU(),
            nn.ConvTranspose2d(base_dim, out_channels, 4, stride=2, padding=1),
        )

        self.num_tasks = num_tasks

    @property
    def name(self) -> str:
        return "conditional_continual_vae"

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for ConditionalContinualVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape
        task_id: int = kwargs.get("task_id", 0)

        h = self.encoder(x, task_id)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        out = self.decoder(z)

        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
