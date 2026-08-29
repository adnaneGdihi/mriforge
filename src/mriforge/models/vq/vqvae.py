"""Vector Quantized Variational Autoencoder (VQ-VAE)

This module implements VQ-VAE for learning discrete latent representations
of MRI data with improved reconstruction quality.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

from .quantizer import VQQuantizer


@register_model(
    name="vqvae",
    training_mode="vae",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class VQVAE(nn.Module, IGenerator):
    """Vector Quantized Variational Autoencoder for MRI super-resolution.

    This model learns discrete latent representations using vector
    quantization, providing better reconstruction quality than standard VAEs.

    Architecture:
    - Encoder: Convolutional network mapping input to latent space
    - Quantizer: VQ layer with EMA codebook updates
    - Decoder: Transpose convolutional network reconstructing from
      quantized latents

    Reference:
    van den Oord et al. "Neural Discrete Representation Learning" NeurIPS 2017
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hiddein_channels: int = 128,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        num_layers: int = 4,
        downsample_factor: int = 4,
        spatial_dims: int = 2,  # 2 for 2D, 3 for 3D
    ):
        """Initialize VQ-VAE.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            hiddein_channels: Number of hidden channels in encoder/decoder
            num_embeddings: Number of codebook entries
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
            num_layers: Number of encoder/decoder layers
            downsample_factor: Total downsampling factor
            spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D)

        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hiddein_channels = hiddein_channels
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.spatial_dims = spatial_dims

        # Encoder
        self.encoder = self._build_encoder(
            in_channels,
            hiddein_channels,
            num_layers,
            downsample_factor,
        )

        # Quantizer
        self.quantizer = VQQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )

        # Decoder
        self.decoder = self._build_decoder(
            embedding_dim,
            hiddein_channels,
            out_channels,
            num_layers,
            downsample_factor,
        )

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "vqvae"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    def _build_encoder(
        self,
        in_ch: int,
        hidden_ch: int,
        num_layers: int,
        downsample_factor: int,
    ) -> nn.Sequential:
        """Build encoder network."""
        layers = []

        # Initial convolution - use appropriate dimensionality
        if self.spatial_dims == 2:
            conv_layer = nn.Conv2d
        elif self.spatial_dims == 3:
            conv_layer = nn.Conv3d
        else:
            raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

        layers.append(conv_layer(in_ch, hidden_ch, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Downsampling layers
        current_ch = hidden_ch
        for _i in range(num_layers - 1):
            next_ch = min(current_ch * 2, hidden_ch * 4)
            layers.append(
                conv_layer(current_ch, next_ch, kernel_size=4, stride=2, padding=1),
            )
            layers.append(nn.ReLU(inplace=True))
            current_ch = next_ch

        # Final convolution to embedding dimension
        layers.append(
            conv_layer(
                current_ch,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        embedding_dim: int,
        hidden_ch: int,
        out_ch: int,
        num_layers: int,
        downsample_factor: int,
    ) -> nn.Sequential:
        """Build decoder network."""
        layers = []

        # Choose appropriate convolution layers
        if self.spatial_dims == 2:
            conv_layer = nn.Conv2d
            conv_transpose_layer = nn.ConvTranspose2d
        elif self.spatial_dims == 3:
            conv_layer = nn.Conv3d
            conv_transpose_layer = nn.ConvTranspose3d
        else:
            raise ValueError(f"Unsupported spatial_dims: {self.spatial_dims}")

        # Initial convolution
        current_ch = hidden_ch * 4  # Start with higher channels
        layers.append(
            conv_layer(embedding_dim, current_ch, kernel_size=3, stride=1, padding=1),
        )
        layers.append(nn.ReLU(inplace=True))

        # Upsampling layers
        for _i in range(num_layers - 1):
            next_ch = current_ch // 2
            layers.append(
                conv_transpose_layer(
                    current_ch,
                    next_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
            layers.append(nn.ReLU(inplace=True))
            current_ch = next_ch

        # Final upsampling and output
        layers.append(
            conv_transpose_layer(
                current_ch,
                hidden_ch,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_layer(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())  # Output in [-1, 1] range

        return nn.Sequential(*layers)

    def _init_weights(self, module: nn.Module):
        """Initialize network weights."""
        # Weight initialization handled by centralized weight_initialization module
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Latent tensor of shape (B, embedding_dim, H', W')

        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to reconstruction.

        Args:
            z: Latent tensor of shape (B, embedding_dim, H', W')

        Returns:
            Reconstructed tensor of shape (B, out_channels, H, W)

        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through VQ-VAE.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or
               (B, C, H, W) for 2D

        Returns:
            Reconstructed tensor of same shape as input

        """
        # Encode
        z_e = self.encode(x)

        # Quantize
        z_q, _, _ = self.quantizer(z_e)

        # Decode
        reconstruction = self.decode(z_q)

        return reconstruction

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space.

        Args:
            z: Latent tensor of shape (B, embedding_dim, H', W')
            **kwargs: Additional generation parameters

        Returns:
            Generated tensor of shape (B, out_channels, H, W)

        """
        return self.decode(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape.

        Args:
            input_shape: Input tensor shape (B, C, H, W)

        Returns:
            Output tensor shape (B, out_channels, H, W)

        """
        # For VQ-VAE, output shape is same as input shape
        # except for the channel dimension
        batch_size, _, height, width = input_shape
        return (batch_size, self.out_channels, height, width)

    def get_quantizer(self) -> VQQuantizer:
        """Get the VQ quantizer for external access."""
        return self.quantizer

    def get_codebook_usage(self) -> torch.Tensor:
        """Get codebook usage statistics."""
        return self.quantizer.get_codebook_usage()

    def get_codebook_entropy(self) -> float:
        """Get codebook entropy."""
        return self.quantizer.get_codebook_entropy()

    def sample_from_codebook(
        self,
        num_samples: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Sample random vectors from the codebook.

        Args:
            num_samples: Number of samples to generate
            device: Device to place samples on

        Returns:
            Sampled latent vectors

        """
        # Sample random indices
        indices = torch.randint(0, self.num_embeddings, (num_samples,), device=device)

        # Convert to embeddings
        samples = self.quantizer.quantize_indices(indices)

        return samples

    def reconstruct_from_indices(
        self,
        indices: torch.Tensor,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        """Reconstruct image from codebook indices.

        Args:
            indices: Codebook indices of shape (B, H', W')
            spatial_size: Spatial dimensions (H, W) for upsampling

        Returns:
            Reconstructed images

        """
        # Convert indices to quantized vectors
        z_q = self.quantizer.quantize_indices(indices.view(-1))
        z_q = z_q.view(
            indices.shape[0],
            indices.shape[1],
            indices.shape[2],
            self.embedding_dim,
        )

        # Transpose to (B, C, H, W) format expected by decoder
        z_q = z_q.permute(0, 3, 1, 2)

        # Upsample if needed
        if z_q.shape[-2:] != spatial_size:
            z_q = F.interpolate(
                z_q,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )

        # Decode
        reconstruction = self.decode(z_q)

        return reconstruction
