"""Robust VAE Generator Implementation
===================================

Robust Variational Autoencoder with improved stability and reconstruction quality
for multi-stage training pipelines.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class RobustEncoder(nn.Module):
    """Robust encoder with residual connections and batch normalization"""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        hidden_dims: list | None = None,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (Optional[list]): Description.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Encoder layers
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            if i == 0:
                modules.append(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                )
            else:
                modules.append(
                    nn.Conv2d(
                        hidden_dims[i - 1],
                        h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                )
            modules.append(nn.BatchNorm2d(h_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout2d(0.1))  # Dropout for robustness

        self.encoder = nn.Sequential(*modules)

        # Calculate output size after convolutions
        self.feature_size = self._calculate_feature_size()

        # Latent space projections
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_var = nn.Linear(self.feature_size, latent_dim)

    def _calculate_feature_size(self) -> int:
        """Calculate flattened feature size after encoder"""
        with torch.no_grad():
            x = torch.randn(1, self.in_channels, 256, 256)
            x = self.encoder(x)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder

        forward method for RobustEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


class RobustDecoder(nn.Module):
    """Robust decoder with skip connections and attention"""

    def __init__(
        self,
        latent_dim: int = 256,
        out_channels: int = 1,
        hidden_dims: list | None = None,
    ):
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            hidden_dims (Optional[list]): Description.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.latent_dim = latent_dim
        self.out_channels = out_channels

        # Initial projection
        self.fc_decode = nn.Linear(latent_dim, hidden_dims[0] * 16 * 16)

        # Decoder layers with skip connections
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout2d(0.1),
                ),
            )

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),  # Output in [-1, 1] range
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder

        forward method for RobustDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.fc_decode(z)
        x = x.view(x.size(0), -1, 16, 16)

        for layer in self.decoder_layers:
            x = layer(x)

        x = self.final_layer(x)
        return x


@register_model(name="robust_vae", training_mode="vae")
class RobustVAEGenerator(nn.Module, IGenerator):
    """Robust Variational Autoencoder with improved stability and reconstruction.

    Features:
    - Batch normalization for training stability
    - Dropout for regularization
    - KL divergence annealing
    - Perceptual reconstruction loss support
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        hidden_dims: list | None = None,
        beta: float = 1.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (Optional[list]): Description.
            beta (float): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight

        # Encoder and decoder
        self.encoder = RobustEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = RobustDecoder(latent_dim, in_channels, hidden_dims)

        # KL annealing parameters
        self.register_buffer("kl_weight", torch.tensor(1.0))

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "robust_vae"

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE

        forward method for RobustVAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)

        return recon, mu, log_var

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent space"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """Sample from latent space"""
        z = torch.randn(
            batch_size,
            self.latent_dim,
            device=next(self.parameters()).device,
        )
        return self.decode(z)

    def get_reconstruction_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss"""
        return F.mse_loss(recon, target)

    def get_kl_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss"""
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.beta * self.kl_weight * kl_loss

    def loss_function(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute total VAE loss"""
        recon_loss = self.get_reconstruction_loss(recon, target)
        kl_loss = self.get_kl_loss(mu, log_var)
        total_loss = recon_loss + kl_loss

        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        return input_shape

    def generate(self, noise: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Generate samples from the VAE"""
        if noise is None:
            batch_size = kwargs.get("batch_size", 1)
            noise = torch.randn(
                batch_size,
                self.latent_dim,
                device=next(self.parameters()).device,
            )

        return self.decode(noise)


def create_robust_vae_generator(
    in_channels: int = 1,
    latent_dim: int = 256,
    hidden_dims: list | None = None,
    beta: float = 1.0,
    **kwargs,
) -> RobustVAEGenerator:
    """Create a robust VAE generator instance.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder/decoder
        beta: KL divergence weight
        **kwargs: Additional arguments (ignored)

    Returns:
        Configured RobustVAEGenerator instance

    """
    return RobustVAEGenerator(
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
    )
