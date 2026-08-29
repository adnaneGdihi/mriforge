"""Scanner-Invariant Embedding Generator

Learns domain-invariant representations across different MRI scanners using
adversarial training and domain adaptation techniques.
"""

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

__all__ = ["DomainDiscriminator", "ScannerInvariantEmbedding"]


class DomainDiscriminator(nn.Module):
    """Discriminator that predicts scanner/domain from features."""

    def __init__(self, feature_dim: int = 256, num_domains: int = 3):
        """__init__.

        Args:
            feature_dim (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling if input is spatial
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for DomainDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.dim() == 4:
            x = torch.mean(x, dim=[2, 3])
        return self.network(x)


@register_model(name="scanner_invariant_embedding", training_mode="domain_adaptation")
class ScannerInvariantEmbedding(nn.Module, IGenerator):
    """Generator with scanner-invariant feature learning.

    Uses adversarial domain adaptation to learn representations that are
    invariant to scanner manufacturer, field strength, and protocol differences.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
        embedding_dim: int = 256,
        num_domains: int = 3,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_features (int): Description.
            embedding_dim (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim

        # Encoder: Extract scanner-invariant features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, 1, 1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, base_features * 2, 3, 2, 1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 2, base_features * 4, 3, 2, 1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features * 4, embedding_dim, 3, 1, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Domain discriminator (for adversarial training)
        self.domain_discriminator = DomainDiscriminator(embedding_dim, num_domains)

        # Decoder: Reconstruct from invariant features
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, base_features * 4, 3, 1, 1),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_features * 4, base_features * 2, 3, 1, 1),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_features * 2, base_features, 3, 1, 1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_features, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract scanner-invariant features."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from features."""
        return self.decoder(z)

    def predict_domain(self, features: torch.Tensor) -> torch.Tensor:
        """Predict domain/scanner from features (for adversarial loss)."""
        return self.domain_discriminator(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode.

        forward method for ScannerInvariantEmbedding.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        z = self.encode(x)
        return self.decode(z)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for ScannerInvariantEmbedding, z is the input state."""
        return self.forward(z)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "scanner_invariant_embedding"

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
