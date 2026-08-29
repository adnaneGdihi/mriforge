"""Causal Discovery Generator for structure-aware MRI reconstruction.

Implements a Causal VAE that learns a directed acyclic graph (DAG) over
latent tissue variables (ρ, T1, T2, B0, etc.) using differentiable
acyclicity constraints. The structural equation model enables causal
interventions on individual tissue properties.

Reference:
    Yang et al., "CausalVAE: Disentangled Representation Learning via
    Neural Structural Causal Models", CVPR 2021.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _DAGLayer(nn.Module):
    """Differentiable DAG adjacency with acyclicity penalty."""

    def __init__(self, num_variables: int) -> None:
        super().__init__()
        self.num_variables = num_variables
        # Learnable adjacency (unconstrained — acyclicity enforced via loss)
        self.W = nn.Parameter(torch.zeros(num_variables, num_variables))

    @property
    def adjacency(self) -> torch.Tensor:
        """Soft adjacency after sigmoid (masking self-loops)."""
        mask = 1.0 - torch.eye(self.num_variables, device=self.W.device)
        return torch.sigmoid(self.W) * mask

    def dag_penalty(self) -> torch.Tensor:
        """NOTEARS-style acyclicity constraint: tr(e^(A∘A)) - d."""
        A = self.adjacency
        M = torch.matrix_exp(A * A)
        return torch.trace(M) - self.num_variables

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply structural equations: z' = A^T @ z + noise.

        forward method for _DAGLayer.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        A = self.adjacency
        return torch.einsum("bid,ij->bjd", z, A) + z  # residual + causal mixing


@register_model(name="causal_discovery", training_mode="reconstruction")
class CausalDiscoveryGenerator(IGenerator, nn.Module):
    """Causal VAE for disentangled MRI tissue property discovery."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_variables: int = kwargs.get("num_variables", 6)
        latent_dim: int = kwargs.get("latent_dim", 16)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 3, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim * 2, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, feature_dim * 2), feature_dim * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(8),
        )
        enc_flat = feature_dim * 2 * 64
        self.fc_mu = nn.Linear(enc_flat, num_variables * latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, num_variables * latent_dim)

        self.num_variables = num_variables
        self.latent_dim = latent_dim

        # DAG structural layer
        self.dag = _DAGLayer(num_variables)

        # Decoder
        self.fc_dec = nn.Linear(num_variables * latent_dim, feature_dim * 2 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, out_channels, 3, padding=1),
        )

    @property
    def name(self) -> str:
        return "causal_discovery"

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for CausalDiscoveryGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B = x.shape[0]
        H, W = x.shape[2], x.shape[3]

        h = self.encoder(x)
        h_flat = h.reshape(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self._reparameterize(mu, logvar)

        # Reshape to [B, num_vars, latent_dim] and apply DAG mixing
        z_vars = z.reshape(B, self.num_variables, self.latent_dim)
        z_causal = self.dag(z_vars)
        z_flat = z_causal.reshape(B, -1)

        h_dec = self.fc_dec(z_flat).reshape(B, -1, 8, 8)
        out = self.decoder(h_dec)

        # Interpolate to match input spatial dims
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
