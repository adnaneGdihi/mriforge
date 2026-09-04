"""Variational Autoencoder (VAE) Generator
=======================================

This module implements a complete Variational Autoencoder for
MRI super-resolution. VAEs learn a probabilistic latent space
representation and can generate high-quality reconstructions.

Architecture:
- Encoder: Maps input images to latent distribution (mean + logvar)
- Decoder: Reconstructs images from latent samples
- Reparameterization: Enables gradient flow through stochastic sampling

Reference:
Kingma et al. "Auto-Encoding Variational Bayes" ICLR 2014
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.latent_access import SupportsLatentAccess
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@dataclass
class VAEGeneratorConfig:
    """Configuration for VAE Generator parameters."""

    in_channels: int = 1
    out_channels: int = 1
    latent_dim: int = 256
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256)
    input_size: int = 64
    beta: float = 1.0  # KL weight for β-VAE
    use_batch_norm: bool = True
    dropout_rate: float = 0.0


class VAEEncoder(nn.Module):
    """VAE Encoder that maps inputs to latent distribution parameters."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 256,
        hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
        input_size: int = 64,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            input_size (int): Description.
            use_batch_norm (bool): Description.
            dropout_rate (float): Description.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.input_size = input_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build encoder layers
        layers = []
        current_channels = in_channels

        for h_dim in hidden_dims:
            layer_list = [
                nn.Conv2d(current_channels, h_dim, kernel_size=3, stride=2, padding=1),
            ]
            if self.use_batch_norm:
                layer_list.append(nn.BatchNorm2d(h_dim))
            layer_list.append(nn.ReLU(inplace=True))
            if self.dropout_rate > 0:
                layer_list.append(nn.Dropout2d(self.dropout_rate))
            layers.extend(layer_list)
            current_channels = h_dim

        self.encoder = nn.Sequential(*layers)
        self.hidden_dims = hidden_dims

        # Defer linear layer creation until we know the actual flattened size
        self.fc_mu = None
        self.fc_logvar = None
        self._flattened_size = None

    def _calculate_flatten_size(
        self,
        in_channels: int,
        hidden_dims: tuple[int, ...],
    ) -> int:
        """Calculate the flattened size after encoder."""
        # Calculate actual output size after conv layers
        current_size = self.input_size
        for _ in hidden_dims:
            # Conv2d with kernel_size=3, stride=2, padding=1
            current_size = (current_size + 2 * 1 - 3) // 2 + 1

        return hidden_dims[-1] * (current_size**2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        forward method for VAEEncoder.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        # Create linear layers on first forward pass if not already created
        if self.fc_mu is None:
            self._flattened_size = h.shape[1]
            self.fc_mu = nn.Linear(self._flattened_size, self.latent_dim)
            self.fc_logvar = nn.Linear(self._flattened_size, self.latent_dim)
            # Move to same device as input
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for gradient flow."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """VAE Decoder that reconstructs images from latent space."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_dim: int = 256,
        hidden_dims: tuple[int, ...] = (256, 128, 64, 32),
        output_size: int = 64,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
    ):
        """__init__.

        Args:
            out_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (tuple[int, ...]): Description.
            output_size (int): Description.
            use_batch_norm (bool): Description.
            dropout_rate (float): Description.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Calculate initial size for decoder
        # With len(hidden_dims) transposed conv layers of stride 2,
        # initial_size = output_size / (2^(num_transposed_convs))
        num_transposed_convs = len(hidden_dims)  # fc + transposed convs
        self.initial_size = output_size // (2**num_transposed_convs)

        # Initial projection
        self.fc_decode = nn.Linear(latent_dim, hidden_dims[0] * (self.initial_size**2))

        # Build decoder layers
        layers = []
        current_channels = hidden_dims[0]

        for _i, h_dim in enumerate(hidden_dims[1:]):
            layer_list = [
                nn.ConvTranspose2d(
                    current_channels,
                    h_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ]
            if self.use_batch_norm:
                layer_list.append(nn.BatchNorm2d(h_dim))
            layer_list.append(nn.ReLU(inplace=True))
            if self.dropout_rate > 0:
                layer_list.append(nn.Dropout2d(self.dropout_rate))
            layers.extend(layer_list)
            current_channels = h_dim

        # Final layer
        layers.extend(
            [
                nn.ConvTranspose2d(
                    current_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.Tanh(),  # Output in [-1, 1] range
            ],
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        forward method for VAEDecoder.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), -1, self.initial_size, self.initial_size)
        return self.decoder(h)


@register_model(name="vae_generator", training_mode="vae")
class VAEGenerator(nn.Module, IGenerator, SupportsLatentAccess):
    """Complete Variational Autoencoder for MRI super-resolution.

    This implementation provides:
    - Probabilistic latent space learning
    - Reconstruction with KL regularization
    - Configurable architecture for different scales
    - Support for conditional generation
    """

    def __init__(
        self,
        config: VAEGeneratorConfig | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        latent_dim: int | None = None,
        hidden_dims: tuple[int, ...] | None = None,
        input_size: int | None = None,
        beta: float | None = None,
        use_batch_norm: bool | None = None,
        dropout_rate: float | None = None,
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        """Initialize VAE Generator.

        Args:
            config: Configuration object with all parameters. If provided,
                   individual parameters are ignored.
            in_channels: Number of input channels (ignored if config provided)
            out_channels: Number of output channels (ignored if config provided)
            latent_dim: Dimension of latent space (ignored if config provided)
            hidden_dims: Hidden dimensions for encoder/decoder (ignored if config provided)
            input_size: Input image size (ignored if config provided)
            beta: KL weight for β-VAE (ignored if config provided)
            use_batch_norm: Whether to use batch normalization (ignored if config provided)
            dropout_rate: Dropout rate (ignored if config provided)
            **kwargs: Additional parameters ignored for compatibility

        """
        super().__init__()

        # Use config if provided, otherwise create from individual parameters
        if config is not None:
            # [FIX] Handle generic ModelConfigSchema from configuration loader
            # usage: factory creates model and passes ModelConfigSchema as 'config'
            if hasattr(config, "model_kwargs"):
                # Extract parameters from ModelConfigSchema and its model_kwargs
                model_kwargs = config.model_kwargs

                self.config = VAEGeneratorConfig(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    latent_dim=model_kwargs.get(
                        "latent_dim", latent_dim if latent_dim is not None else 256
                    ),
                    hidden_dims=tuple(model_kwargs.get("hidden_dims", (32, 64, 128, 256))),
                    input_size=model_kwargs.get("input_size", 64),
                    beta=model_kwargs.get("beta", 1.0),
                    use_batch_norm=model_kwargs.get("use_batch_norm", True),
                    dropout_rate=model_kwargs.get("dropout_rate", 0.0),
                )
            else:
                self.config = config
        else:
            self.config = VAEGeneratorConfig(
                in_channels=in_channels if in_channels is not None else 1,
                out_channels=out_channels if out_channels is not None else 1,
                latent_dim=latent_dim if latent_dim is not None else 256,
                hidden_dims=(hidden_dims if hidden_dims is not None else (32, 64, 128, 256)),
                input_size=input_size if input_size is not None else 64,
                beta=beta if beta is not None else 1.0,
                use_batch_norm=use_batch_norm if use_batch_norm is not None else True,
                dropout_rate=dropout_rate if dropout_rate is not None else 0.0,
            )

        # Set instance attributes from config
        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.latent_dim = self.config.latent_dim
        self.input_size = self.config.input_size
        self.beta = self.config.beta
        self.use_batch_norm = self.config.use_batch_norm
        self.dropout_rate = self.config.dropout_rate

        # Encoder and Decoder
        self.encoder = VAEEncoder(
            in_channels=self.in_channels,
            latent_dim=self.latent_dim,
            hidden_dims=self.config.hidden_dims,
            input_size=self.input_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )

        self.decoder = VAEDecoder(
            out_channels=self.out_channels,
            latent_dim=self.latent_dim,
            hidden_dims=tuple(reversed(self.config.hidden_dims)),
            output_size=self.input_size,  # Decoder outputs same size as encoder input
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "vae_sr"

    # [PI-FIX] Implement SupportsLatentAccess for robust multi-stage handoff
    def export_latent_state(self) -> dict[str, torch.Tensor]:
        """Export internal latent state."""
        # Caller is expected to invoke ``forward`` first; this method
        # reads the cached (mu, logvar) from the last forward pass.
        #
        # Warning: This method assumes we want to re-encode the *last input* or are
        # being used in a context where we can get the state.
        # Actually, `EncoderDecoderStageModel` calls `export_latent_state` on the ENCODER.
        # `VAEGenerator` is the whole model.
        #
        # If VAEGenerator is used as the 'Encoder' in the stage config (unlikely for VAE),
        # but if it is used as 'Model', the service handles the tuple return.
        #
        # Wait, the PI report says: "The multi-stage service's extraction logic should explicitly prioritize this method".
        # So we should provide a way to get the latent FROM the model instance.
        #
        # Implementation: cache last latents during forward?
        if hasattr(self, "_last_mu") and hasattr(self, "_last_logvar"):
            z = self.reparameterize(self._last_mu, self._last_logvar)
            return {
                "embedding": z,
                "mu": self._last_mu,
                "logvar": self._last_logvar,
                "z": z,
            }
        return {}

    def get_latent_embedding(self) -> torch.Tensor | None:
        """Get the primary latent embedding (z)."""
        if hasattr(self, "_last_mu") and hasattr(self, "_last_logvar"):
            return self.reparameterize(self._last_mu, self._last_logvar)
        return None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (reconstructed_tensor, mu, logvar) for loss computation

        forward method for VAEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [SAFETY] Ensure input is 4D (B, C, H, W)
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # Encode
        mu, logvar = self.encoder(x)

        # [PI-FIX] Cache latents for export
        self._last_mu = mu
        self._last_logvar = logvar

        # Reparameterize
        z = self.encoder.reparameterize(mu, logvar)

        # Decode
        reconstruction = self.decoder(z)

        # Resize to match input size if necessary
        if reconstruction.shape[-2:] != x.shape[-2:]:
            reconstruction = torch.nn.functional.interpolate(
                reconstruction,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return reconstruction, mu, logvar

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        return self.encoder.reparameterize(mu, logvar)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space.

        Args:
            z: Latent vectors [B, latent_dim]
            **kwargs: Additional generation parameters

        Returns:
            Generated images [B, out_channels, H, W]

        """
        return self.decode(z)

    def sample(self, num_samples: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample random latent vectors and generate images."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generate(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        batch_size = input_shape[0]
        return (batch_size, self.out_channels, self.input_size, self.input_size)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def loss_function(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss with reconstruction and KL terms.

        Args:
            reconstruction: Reconstructed images
            target: Target images
            mu: Latent mean
            logvar: Latent log variance
            beta: KL weight (overrides self.beta if provided)

        Returns:
            Dictionary with loss components

        """
        beta = beta if beta is not None else self.beta

        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(reconstruction, target, reduction="sum")

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }


def create_vae_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    latent_dim: int = 256,
    input_size: int = 64,
    beta: float = 1.0,
    hidden_dims: tuple[int, ...] = (32, 64, 128, 256),
    use_batch_norm: bool = True,
    dropout_rate: float = 0.0,
    **kwargs,
) -> VAEGenerator:
    """Factory function for creating VAE generators.

    Args:
        in_channels: Number of input/output channels
        out_channels: Number of output classes (unused for VAE)
        latent_dim: Dimension of latent space
        input_size: Input image size (assumed square)
        beta: KL divergence weight
        hidden_dims: Hidden dimensions for encoder/decoder
        use_batch_norm: Whether to use batch normalization
        dropout_rate: Dropout rate
        **kwargs: Additional arguments passed to VAEGenerator

    Returns:
        Configured VAE generator

    """
    return VAEGenerator(
        in_channels=in_channels,
        out_channels=in_channels,
        latent_dim=latent_dim,
        input_size=input_size,
        beta=beta,
        hidden_dims=hidden_dims,
        use_batch_norm=use_batch_norm,
        dropout_rate=dropout_rate,
        **kwargs,
    )
