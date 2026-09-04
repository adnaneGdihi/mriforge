"""Sparse Variational Autoencoder (Sparse VAE) Generator
======================================================

This module implements a Sparse Variational Autoencoder for
MRI super-resolution. Sparse VAEs encourage sparsity in the latent
representation through learned pruning and sparsity regularization.

Architecture:
- Encoder: Maps input images to latent distribution (mean + logvar)
- Decoder: Reconstructs images from latent samples
- Sparsity Regularization: Encourages sparse latent representations
- Reparameterization: Enables gradient flow through stochastic sampling

Features:
- Learned sparsity mask via straight-through estimator
- Configurable sparsity target and regularization strength
- Low KL weight (β) to focus on reconstruction
- Sparsity statistics tracking
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

from ..vae import create_sparse_vae


@dataclass
class SparseVAEGeneratorConfig:
    """Configuration for Sparse VAE Generator parameters."""

    in_channels: int = 1
    out_channels: int = 1
    latent_dim: int = 256
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256)
    beta_kl: float = 0.001  # Weak KL weight for sparse VAE
    sparsity_lambda: float = 0.001  # Sparsity regularization weight
    sparsity_target: float = 0.05  # Target sparsity (fraction of zeros)
    use_batch_norm: bool = True
    dropout_rate: float = 0.0


@register_model(name="sparse_vae_generator", training_mode="vae")
class SparseVAEGenerator(nn.Module, IGenerator):
    """Sparse VAE Generator for MRI super-resolution.

    Combines VAE architecture with sparsity regularization to learn
    compact and interpretable latent representations.

    Features:
    - Learned sparsity mask for pruning latent dimensions
    - Configurable sparsity target
    - Weak KL divergence weight to focus on reconstruction
    - Sparsity statistics for analysis
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 256,
        hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
        beta_kl: float = 0.001,
        sparsity_lambda: float = 0.001,
        sparsity_target: float = 0.05,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        """Initialize Sparse VAE Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            latent_dim: Dimension of latent space
            hidden_dims: Hidden dimensions for encoder/decoder
            beta_kl: Weight for KL divergence loss (low for sparse VAE)
            sparsity_lambda: Weight for sparsity regularization
            sparsity_target: Target sparsity level (fraction of zeros)
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta_kl = beta_kl
        self.sparsity_lambda = sparsity_lambda
        self.sparsity_target = sparsity_target

        # Create sparse VAE model
        self.vae = create_sparse_vae(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            beta_kl=beta_kl,
            sparsity_lambda=sparsity_lambda,
            sparsity_target=sparsity_target,
        )

        # Loss tracking
        self._last_kl_loss = None
        self._last_sparsity_loss = None
        self._last_recon_loss = None

    def forward(
        self,
        x: torch.Tensor,
        *_extra_pos: Any,
        apply_sparsity: bool = True,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through Sparse VAE Generator.

        Args:
            x: Input tensor [B, C, H, W]
            apply_sparsity: Whether to apply sparsity mask

        Returns:
            Reconstructed tensor [B, C, H, W]

        forward method for SparseVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            apply_sparsity (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        reconstruction, _, _, _ = self.vae.forward(x, apply_sparsity=apply_sparsity)
        if reconstruction.shape[2:] != x.shape[2:]:
            reconstruction = F.interpolate(
                reconstruction, size=x.shape[2:], mode="bilinear", align_corners=False
            )
        return reconstruction

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (mu, log_var) for latent distribution

        """
        return self.vae.encode(x)

    def decode(
        self,
        z: torch.Tensor,
        apply_sparsity: bool = True,
    ) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector [B, latent_dim]
            apply_sparsity: Whether to apply sparsity mask

        Returns:
            Reconstructed tensor [B, C, H, W]

        """
        return self.vae.decode(z, apply_sparsity=apply_sparsity)

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
        return self.vae.sample(
            num_samples=num_samples,
            device=device,
            apply_sparsity=apply_sparsity,
        )

    def reconstruct(
        self,
        x: torch.Tensor,
        apply_sparsity: bool = True,
    ) -> torch.Tensor:
        """Reconstruct input through Sparse VAE.

        Args:
            x: Input tensor [B, C, H, W]
            apply_sparsity: Whether to apply sparsity mask

        Returns:
            Reconstructed tensor [B, C, H, W]

        """
        return self.vae.reconstruct(x, apply_sparsity=apply_sparsity)

    def compute_loss(
        self,
        x: torch.Tensor,
        recon_loss_fn: nn.Module | None = None,
        apply_sparsity: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Compute total loss for Sparse VAE.

        Args:
            x: Input tensor [B, C, H, W]
            recon_loss_fn: Reconstruction loss function (default: MSE)
            apply_sparsity: Whether to apply sparsity regularization

        Returns:
            Dictionary with loss components

        """
        if recon_loss_fn is None:
            recon_loss_fn = F.mse_loss

        # Forward pass
        reconstruction, mu, log_var, z = self.vae.forward(x, apply_sparsity=True)

        # Reconstruction loss
        recon_loss = recon_loss_fn(reconstruction, x)
        self._last_recon_loss = recon_loss.item()

        # KL divergence loss
        kl_loss = self.vae.compute_kl_loss(mu, log_var)
        self._last_kl_loss = kl_loss.item()

        # Sparsity loss
        sparsity_loss = torch.tensor(0.0, device=x.device)
        if apply_sparsity and z is not None:
            sparsity_loss = self.vae.compute_sparsity_loss(z)
            self._last_sparsity_loss = sparsity_loss.item()

        # Total loss
        total_loss = recon_loss + kl_loss + sparsity_loss

        return {
            "total": total_loss,
            "reconstruction": recon_loss,
            "kl_divergence": kl_loss,
            "sparsity": sparsity_loss,
        }

    def get_sparsity_stats(self, z: torch.Tensor) -> dict[str, float]:
        """Get sparsity statistics for latent representation.

        Args:
            z: Latent vector [B, latent_dim]

        Returns:
            Dictionary with sparsity statistics

        """
        return self.vae.get_sparsity_stats(z)

    def update_sparsity_mask(self, z: torch.Tensor, threshold: float = 0.01):
        """Update learned sparsity mask based on latent statistics.

        Args:
            z: Latent vector [B, latent_dim]
            threshold: Threshold for pruning dimensions

        """
        self.vae.update_sparsity_mask(z, threshold=threshold)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def generate(
        self,
        z: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Generate samples from latent vectors (IGenerator interface).

        Args:
            z: Latent vectors [B, latent_dim]
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            Generated samples [B, out_channels, H, W]

        """
        return self.decode(z, apply_sparsity=kwargs.get("apply_sparsity", True))

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Returns the output shape for a given input shape (IGenerator interface).

        Args:
            input_shape: Input shape tuple (B, C, H, W)

        Returns:
            Output shape tuple (B, out_channels, H, W)

        """
        if len(input_shape) == 4:
            return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])
        elif len(input_shape) == 2:
            # Latent space shape - decoder outputs (B, C, H, W)
            # Assume standard 64x64 output
            return (input_shape[0], self.out_channels, 64, 64)
        else:
            raise ValueError(f"Unsupported input_shape: {input_shape}")

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "SparseVAEGenerator"


def create_sparse_vae_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 256,
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
    beta_kl: float = 0.001,
    sparsity_lambda: float = 0.001,
    sparsity_target: float = 0.05,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
    **kwargs,
) -> SparseVAEGenerator:
    """Factory function for creating Sparse VAE Generators.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder/decoder
        beta_kl: Weight for KL divergence loss
        sparsity_lambda: Weight for sparsity regularization
        sparsity_target: Target sparsity level
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate for regularization

    Returns:
        Configured Sparse VAE Generator

    """
    return SparseVAEGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta_kl=beta_kl,
        sparsity_lambda=sparsity_lambda,
        sparsity_target=sparsity_target,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        **kwargs,
    )
