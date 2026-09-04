"""3D VAE Generator Implementation
==============================

3D Variational Autoencoder for volumetric latent space learning
in multi-stage training pipelines.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class Encoder3D(nn.Module):
    """3D encoder with 3D convolutions for volumetric data"""

    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 512,
        hidden_dims: list | None = None,
        input_shape: tuple[int, int, int] = (64, 64, 64),
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (Optional[list]): Description.
            input_shape (tuple[int, int, int]): Description.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder layers
        modules = []
        for i, h_dim in enumerate(hidden_dims):
            if i == 0:
                modules.append(
                    nn.Conv3d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                )
            else:
                modules.append(
                    nn.Conv3d(
                        hidden_dims[i - 1],
                        h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                )
            modules.append(nn.BatchNorm3d(h_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout3d(0.1))  # Dropout for robustness

        self.encoder = nn.Sequential(*modules)

        # Calculate output size after convolutions
        self.feature_size = self._calculate_feature_size()

        # Latent space projections
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_var = nn.Linear(self.feature_size, latent_dim)

        self.use_checkpoint = True

    def _calculate_feature_size(self) -> int:
        """Calculate flattened feature size after encoder"""
        with torch.no_grad():
            # Use configured input size (D, H, W)
            # Add batch and channel dims: (1, C, D, H, W)
            dummy_input = torch.zeros(1, self.in_channels, *self.input_shape)
            x = self.encoder(dummy_input)
            return x.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through 3D encoder

        forward method for Encoder3D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if getattr(self, "use_checkpoint", False) and self.training:
            x = torch.utils.checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # Get latent parameters
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var


class Decoder3D(nn.Module):
    """3D decoder with transposed 3D convolutions"""

    def __init__(
        self,
        latent_dim: int = 512,
        out_channels: int = 1,
        hidden_dims: list | None = None,
        input_shape: tuple[int, int, int] = (64, 64, 64),
    ):
        """__init__.

        Args:
            latent_dim (int): Description.
            out_channels (int): Description.
            hidden_dims (Optional[list]): Description.
            input_shape (tuple[int, int, int]): Description.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.input_shape = input_shape

        # Calculate bottleneck spatial dimensions
        # Assuming 4 downsampling layers with stride 2
        # If hidden_dims length implies different depth, we should calculate properly
        num_layers = len(hidden_dims)
        # Note: Decoder hidden_dims usually includes the output of the first FC reshape
        # But here logic is: FC -> Reshape -> Deconvs.
        # The number of upsampling layers matches the number of Deconv blocks.

        # Calculate reduction factor based on number of layers defined below
        # The loop runs len(hidden_dims) - 1 times, plus valid final layer.
        # Total upsamples = len(hidden_dims)
        # Factor = 2 ** len(hidden_dims)

        factor = 2 ** len(hidden_dims)
        self.bottleneck_shape = (
            input_shape[0] // factor,
            input_shape[1] // factor,
            input_shape[2] // factor,
        )

        bottleneck_flat = (
            hidden_dims[0]
            * self.bottleneck_shape[0]
            * self.bottleneck_shape[1]
            * self.bottleneck_shape[2]
        )

        # Initial projection
        self.fc_decode = nn.Linear(latent_dim, bottleneck_flat)

        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm3d(hidden_dims[i + 1]),
                    nn.ReLU(),
                    nn.Dropout3d(0.1),
                ),
            )

        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                hidden_dims[-1],
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        self.use_checkpoint = True

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D decoder

        forward method for Decoder3D.

        Executes PyTorch tensor operations.

        Args:
            z (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = self.fc_decode(z)
        # Reshape to bottleneck spatial dims
        # [B, C, D, H, W]
        x = x.view(
            x.size(0),
            -1,
            self.bottleneck_shape[0],
            self.bottleneck_shape[1],
            self.bottleneck_shape[2],
        )

        for layer in self.decoder_layers:
            if getattr(self, "use_checkpoint", False) and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.final_layer(x)
        return x


@register_model(name="vae_3d", training_mode="vae")
class VAE3DGenerator(nn.Module, IGenerator):
    """3D Variational Autoencoder for volumetric data.

    Features:
    - 3D convolutions for volumetric processing
    - Batch normalization for training stability
    - Dropout for regularization
    - KL divergence annealing
    - Support for 3D medical imaging data
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,  # Added for model factory compatibility
        latent_dim: int = 512,
        hidden_dims: list | None = None,
        beta: float = 1.0,
        input_shape: tuple[int, int, int] = (64, 64, 64),  # Added input_shape
        **kwargs,  # Accept additional kwargs for compatibility
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            latent_dim (int): Description.
            hidden_dims (Optional[list]): Description.
            beta (float): Description.
            input_shape (tuple[int, int, int]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight
        self.input_shape = input_shape

        # Encoder and decoder
        self.encoder = Encoder3D(in_channels, latent_dim, hidden_dims, input_shape=input_shape)
        self.decoder = Decoder3D(latent_dim, out_channels, hidden_dims, input_shape=input_shape)

        # KL annealing parameters
        self.register_buffer("kl_weight", torch.tensor(1.0))

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "vae_3d"

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        # [FIX] Clamp log_var to prevent numerical explosion
        # exp(10) ~ 22000, exp(20) ~ 4.8e8. Clamp to [-20, 20] is safe.
        log_var = torch.clamp(log_var, max=20.0, min=-20.0)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through 3D VAE

        forward method for VAE3DGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Handle 4D Input from generic pipelines (B, H, W, D) -> (B, C, D, H, W)
        # Expected input shape: (B, C, D, H, W)
        # Received 4D: (B, H, W, D) or (B, C, H, W)?
        # From logs: [1, 64, 64, 32]. Patch size is [64, 64, 32].
        # So it looks like (B, H, W, D) = (1, 64, 64, 32).

        orig_x_shape = x.shape

        if x.dim() == 4:
            # Assume (B, C, H, W) from train.py squeeze
            # Convert to (B, C, 1, H, W)
            x = x.unsqueeze(2)

        # [FIX] Ensure depth is sufficient for 3D Conv (kernel=4) across multiple layers
        # If input has small depth (e.g. 2D slice with D=1), repeat it to at least 16
        if x.dim() == 5 and x.shape[2] < 16:
            repeats = 16 // x.shape[2] + 1
            x = x.repeat(1, 1, repeats, 1, 1)
            x = x[:, :, :16, :, :]

        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)

        # [FIX] Restore original shape if we modified it (e.g. 4D->5D or Depth Repeat)
        if recon.shape != orig_x_shape:
            # Basic Interpolation to match spatial/depth dims
            # Handle dim mismatch (5D recon vs 4D orig)
            target_size = orig_x_shape[2:] if len(orig_x_shape) == 5 else orig_x_shape[2:]
            # If 4D orig (B,C,H,W), target_size is (H,W).
            # But recon is 5D (B,C,D,H,W). We want output to match orig dims.

            if len(orig_x_shape) == 4:
                # We likely need to squeeze D if it's 1, or interpolate D to 1 then squeeze
                # But we repeated D=1 -> D=16. So we should mean or slice?
                # Interpolate to (1, H, W) then squeeze is safest?
                # Or strict interpolate to orig_x_shape size.
                pass  # logic below handles general case

            # Robust Interpolate
            # If recon is 5D and orig is 4D, we treat orig as (B,C,1,H,W) size-wise for interpolation
            # then squeeze.

            if recon.dim() == 5 and len(orig_x_shape) == 4:
                # Interpolate to (D=1, H, W)
                recon = F.interpolate(
                    recon,
                    size=(1, *orig_x_shape[2:]),
                    mode="trilinear",
                    align_corners=False,
                )
                recon = recon.squeeze(2)
            elif recon.dim() == 5 and len(orig_x_shape) == 5:
                recon = F.interpolate(
                    recon, size=orig_x_shape[2:], mode="trilinear", align_corners=False
                )
            else:
                # Fallback
                if recon.shape != orig_x_shape:
                    recon = F.interpolate(
                        recon,
                        size=orig_x_shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

        return recon, mu, log_var

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode 3D input to latent space"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to 3D volume"""
        return self.decoder(z)

    def sample(self, batch_size: int = 1) -> torch.Tensor:
        """Sample from 3D latent space"""
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
        """Compute 3D reconstruction loss"""
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
        """Compute total 3D VAE loss"""
        recon_loss = self.get_reconstruction_loss(recon, target)
        kl_loss = self.get_kl_loss(mu, log_var)
        total_loss = recon_loss + kl_loss

        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given 3D input shape."""
        # Assume 3D input (C, D, H, W) -> output same shape
        return input_shape

    def generate(self, noise: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        """Generate 3D samples from the VAE"""
        if noise is None:
            batch_size = kwargs.get("batch_size", 1)
            noise = torch.randn(
                batch_size,
                self.latent_dim,
                device=next(self.parameters()).device,
            )

        return self.decode(noise)


def create_vae_3d_generator(
    in_channels: int = 1,
    latent_dim: int = 512,
    hidden_dims: list | None = None,
    beta: float = 1.0,
    **kwargs,
) -> VAE3DGenerator:
    """Create a 3D VAE generator instance.

    Args:
        in_channels: Number of input channels
        latent_dim: Dimension of latent space
        hidden_dims: Hidden dimensions for encoder/decoder
        beta: KL divergence weight
        **kwargs: Additional arguments passed to VAE3DGenerator

    Returns:
        Configured VAE3DGenerator instance

    """
    return VAE3DGenerator(
        in_channels=in_channels,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        beta=beta,
    )
