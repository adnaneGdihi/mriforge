"""Hyperspectral GAN for Multi-Contrast MRI Synthesis

Generates multiple MRI contrasts simultaneously with inter-contrast consistency.
"""

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SpectralAttention(nn.Module):
    """Attention mechanism across spectral (contrast) dimension."""

    def __init__(self, num_contrasts: int, feature_dim: int):
        """__init__.

        Args:
            num_contrasts (int): Description.
            feature_dim (int): Description.
        """
        super().__init__()
        self.num_contrasts = num_contrasts
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, num_contrasts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral attention.

        Args:
            x: [B, C, num_contrasts, H, W]

        forward method for SpectralAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, N, H, W = x.shape

        # Global pooling across spatial dims
        pooled = x.mean(dim=[-2, -1])  # [B, C, N]

        # Compute attention weights
        weights = self.attention(pooled.permute(0, 2, 1))  # [B, N, C]
        weights = weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, C, N, 1, 1]

        # Apply attention
        return x * weights


@register_model(name="hyperspectral_gan", training_mode="gan")
class HyperspectralGAN(nn.Module, IGenerator):
    """Multi-contrast GAN for hyperspectral MRI synthesis.

    Generates multiple contrasts (T1, T2, FLAIR, etc.) simultaneously
    with cross-contrast consistency constraints.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_contrasts: int = 3,
        base_features: int = 64,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_contrasts (int): Description.
            base_features (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_contrasts = num_contrasts

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 4, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

        # Spectral attention
        self.spectral_attention = SpectralAttention(num_contrasts, base_features * 4)

        # Contrast-specific decoders
        self.decoders = nn.ModuleList(
            [self._make_decoder(base_features * 4, out_channels) for _ in range(num_contrasts)]
        )

        # Cross-contrast fusion
        self.fusion = nn.Conv2d(out_channels * num_contrasts, out_channels, 1)

    def _make_decoder(self, in_features: int, out_channels: int) -> nn.Module:
        """Create decoder for a single contrast."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_features, in_features // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_features // 2, in_features // 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 4, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward_multi_contrast(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Generate all contrasts separately."""
        # Encode
        features = self.encoder(x)  # [B, C*4, H/4, W/4]

        # Replicate for each contrast
        B, C, H, W = features.shape
        features_expanded = features.unsqueeze(2).repeat(1, 1, self.num_contrasts, 1, 1)

        # Apply spectral attention
        features_attended = self.spectral_attention(features_expanded)

        # Decode each contrast
        contrasts = []
        for i, decoder in enumerate(self.decoders):
            contrast_features = features_attended[:, :, i, :, :]
            contrast = decoder(contrast_features)
            contrasts.append(contrast)

        return contrasts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns fused multi-contrast output.

        forward method for HyperspectralGAN.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        contrasts = self.forward_multi_contrast(x)

        # Concatenate and fuse
        concatenated = torch.cat(contrasts, dim=1)
        fused = self.fusion(concatenated)

        return fused

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for Hyperspectral GAN, z is the input state."""
        return self.forward(z)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "hyperspectral_gan"

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape
