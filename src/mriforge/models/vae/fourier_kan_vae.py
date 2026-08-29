"""FourierKAN VAE
===============

Complete VAE implementation using FourierKAN layers for
frequency domain latent space modeling.
"""

import torch
import torch.nn as nn

from mriforge.models.registry import register_model

from .fourier_kan_decoder import FourierKANDecoder
from .fourier_kan_encoder import FourierKANEncoder


@register_model(name="vae_fourier_kan_vae", training_mode="vae")
class FourierKANVAE(nn.Module):
    """VAE with FourierKAN layers for frequency domain feature learning."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_dim: int = 256,
        num_bases: int = 8,
        max_frequency: float = 1.0,
        basis_trainable: bool = True,
        input_size: int = 64,  # Expected input spatial size
        use_complex_conv: bool = False,
        use_gabor: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            latent_dim (int): Description.
            num_bases (int): Description.
            max_frequency (float): Description.
            basis_trainable (bool): Description.
            input_size (int): Description.
            use_complex_conv (bool): Description.
            use_gabor (bool): Description.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        # Encoder and Decoder
        self.encoder = FourierKANEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )
        self.decoder = FourierKANDecoder(
            out_channels=out_channels,
            latent_dim=latent_dim,
            num_bases=num_bases,
            max_frequency=max_frequency,
            basis_trainable=basis_trainable,
            output_size=input_size,  # Match decoder output to input size
            use_complex_conv=use_complex_conv,
            use_gabor=use_gabor,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encode
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Description.

        forward method for FourierKANVAE.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        mu, logvar = self.encoder(x)

        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def loss_function(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss with reconstruction and KL terms."""
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample from the latent space."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        self.decoder.to(device)
        return self.decoder(z)
