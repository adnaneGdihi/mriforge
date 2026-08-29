#!/usr/bin/env python3
"""Sparse VAE implementation for MRI super-resolution.

This module implements a Sparse Variational Autoencoder that encourages
sparsity in the latent representation through learned pruning and
sparsity regularization.
"""

from typing import Any

import torch
from torch import nn

from mriforge.models.interfaces.models import IModel
from mriforge.models.registry import register_model

from .decoder import VAEDecoder
from .encoder import VAEEncoder


@register_model(name="sparse_vae", training_mode="vae")
class SparseVAE(nn.Module, IModel):
    """Sparse Variational Autoencoder for MRI super-resolution.

    Combines encoder and decoder with reparameterization trick,
    KL divergence loss, and sparsity regularization to encourage
    sparse latent representations.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        beta_kl: float = 0.001,
        sparsity_lambda: float = 0.001,
        sparsity_target: float = 0.05,
    ):
        """Initialize Sparse VAE.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for encoder/decoder
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            beta_kl: Weight for KL divergence loss (default: 0.001 for weak KL)
            sparsity_lambda: Weight for sparsity regularization
            sparsity_target: Target sparsity level (fraction of zeros)

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta_kl = beta_kl
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        # Create encoder and decoder
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        self.decoder = VAEDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            hidden_dims=tuple(reversed(hidden_dims)),
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
        )

        # Learned sparsity mask (learned via straight-through estimator)
        self.register_buffer(
            "sparsity_mask",
            torch.ones(latent_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    def forward(
        self,
        x: torch.Tensor,
        *_extra_pos: Any,
        apply_sparsity: bool = True,
        **_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass through Sparse VAE.

        Args:
            x: Input tensor [B, C, H, W]
            apply_sparsity: Whether to apply sparsity mask

        Returns:
            Tuple of (reconstruction, mu, log_var, sparse_z)

        forward method for SparseVAE.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            apply_sparsity (bool): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Encode
        mu, log_var = self.encoder(x)

        # Reparameterize
        z = self.encoder.reparameterize(mu, log_var)

        # Apply sparsity mask
        sparse_z = None
        if apply_sparsity:
            sparse_z = z * self.sparsity_mask.view(1, -1)
            reconstruction = self.decoder(sparse_z)
        else:
            reconstruction = self.decoder(z)
            sparse_z = z

        # Interpolate to match input spatial dimensions if necessary
        if reconstruction.shape[2:] != x.shape[2:]:
            import torch.nn.functional as F

            reconstruction = F.interpolate(
                reconstruction, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return reconstruction, mu, log_var, sparse_z

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor, apply_sparsity: bool = True) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        if apply_sparsity:
            z = z * self.sparsity_mask.view(1, -1)
        return self.decoder.decode(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        return self.encoder.reparameterize(mu, log_var)

    def sample(
        self,
        num_samples: int = 1,
        device: torch.device | None = None,
        apply_sparsity: bool = True,
    ) -> torch.Tensor:
        """Sample from latent space.

        Args:
            num_samples: Number of samples to generate
            device: Device to place samples on
            apply_sparsity: Whether to apply sparsity mask

        Returns:
            Generated samples [num_samples, out_channels, H, W]

        """
        if device is None:
            device = next(self.parameters()).device

        # Sample from standard normal
        z = torch.randn(num_samples, self.latent_dim, device=device)

        # Decode
        return self.decode(z, apply_sparsity=apply_sparsity)

    def reconstruct(
        self,
        x: torch.Tensor,
        apply_sparsity: bool = True,
    ) -> torch.Tensor:
        """Reconstruct input through Sparse VAE."""
        reconstruction, _, _, _ = self.forward(x, apply_sparsity=apply_sparsity)
        return reconstruction

    def compute_kl_loss(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss.

        Args:
            mu: Mean of latent distribution [B, latent_dim]
            log_var: Log variance of latent distribution [B, latent_dim]

        Returns:
            Weighted KL divergence loss

        """
        # KL divergence: 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return self.beta_kl * torch.mean(kl_loss)

    def compute_sparsity_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparsity regularization loss.

        Encourages the latent representation to have a target sparsity level.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Sparsity regularization loss

        """
        # Target: dimensions with small average magnitude should be sparse
        # Loss encourages low values (sparsity) via L1 norm
        sparsity_loss = torch.mean(torch.abs(z))

        # Alternative: KL divergence between observed and target sparsity
        # This encourages the fraction of near-zero values to match target
        mask_prob = torch.mean((torch.abs(z) < 0.1).float(), dim=0)  # [latent_dim]
        target_prob = self.sparsity_target * torch.ones_like(mask_prob)

        kl_sparse = torch.sum(
            target_prob * torch.log(target_prob / (mask_prob + 1e-8))
            + (1 - target_prob) * torch.log((1 - target_prob) / (1 - mask_prob + 1e-8))
        )

        return self.sparsity_lambda * (sparsity_loss + kl_sparse)

    def get_latent_distribution(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get latent distribution parameters."""
        return self.encode(x)

    def get_sparsity_stats(self, z: torch.Tensor) -> dict[str, float]:
        """Get sparsity statistics for latent representation.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Dictionary with sparsity statistics

        """
        with torch.no_grad():
            z_abs = torch.abs(z)
            sparsity_threshold_01 = (z_abs < 0.1).float().mean().item()
            sparsity_threshold_001 = (z_abs < 0.01).float().mean().item()
            sparsity_threshold_0001 = (z_abs < 0.001).float().mean().item()

            avg_magnitude = z_abs.mean().item()
            max_magnitude = z_abs.max().item()
            min_magnitude = z_abs.min().item()

            return {
                "sparsity_@_0.1": sparsity_threshold_01,
                "sparsity_@_0.01": sparsity_threshold_001,
                "sparsity_@_0.001": sparsity_threshold_0001,
                "avg_magnitude": avg_magnitude,
                "max_magnitude": max_magnitude,
                "min_magnitude": min_magnitude,
            }

    def update_sparsity_mask(self, z: torch.Tensor, threshold: float = 0.01):
        """Update learned sparsity mask based on latent statistics.

        Uses straight-through estimator concept to learn which dimensions
        should be pruned during backpropagation.

        Args:
            z: Latent vector [B, latent_dim]
            threshold: Threshold for pruning dimensions

        """
        with torch.no_grad():
            avg_magnitude = torch.mean(torch.abs(z), dim=0)  # [latent_dim]
            # Mask dimensions below threshold
            self.sparsity_mask = (avg_magnitude > threshold).float()

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "SparseVAE"


def create_sparse_vae(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 128,
    hidden_dims: tuple[int, ...] = (64, 128, 256, 512),
    beta_kl: float = 0.001,
    sparsity_lambda: float = 0.001,
    sparsity_target: float = 0.05,
) -> SparseVAE:
    """Factory function for creating Sparse VAEs.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder/decoder
        beta_kl: Weight for KL divergence loss
        sparsity_lambda: Weight for sparsity regularization
        sparsity_target: Target sparsity level

    Returns:
        Configured Sparse VAE

    """
    return SparseVAE(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta_kl=beta_kl,
        sparsity_lambda=sparsity_lambda,
        sparsity_target=sparsity_target,
    )
