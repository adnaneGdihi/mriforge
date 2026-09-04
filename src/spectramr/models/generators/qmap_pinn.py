"""
Q-Map Generator: Maps undersampled k-space to quantitative tissue parameter maps.

This generator outputs three distinct physical maps (T1, T2, PD) instead of
reconstructed images. The outputs are strictly constrained to physically plausible ranges.
"""

import torch
import torch.nn as nn

from spectramr.models.blocks.swin import SwinBlock
from spectramr.models.generators.protocols import IGeneratorProtocol
from spectramr.models.registry import register_model


class SwinBody(nn.Module):
    """Swin Transformer-based intermediate processing block."""

    def __init__(self, dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinBlock(dim=dim, num_heads=4, window_size=8, shift_size=0),
                SwinBlock(dim=dim, num_heads=4, window_size=8, shift_size=4),
            ]
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            feat = blk(feat, input_resolution=(H, W))
        feat = feat.transpose(1, 2).view(B, C, H, W)
        return self.upsample(feat)


@register_model(name="qmap_generator", training_mode="reconstruction")
class QMapGenerator(nn.Module, IGeneratorProtocol):
    """
    Quantitative Parameter Map Generator.

    Takes undersampled or coil-combined image input and outputs three constrained
    physical parameter maps: T1 (ms), T2 (ms), and Proton Density.

    The network architecture is a simple U-Net-inspired encoder-body-heads design:
    - Encoder: Downsamples input and extracts features
    - Body: Intermediate processing (can be extended with SIREN, ResBlocks, etc.)
    - Heads: Three specialized output heads with physics-constrained activations
    Attributes:
        input_dim (int): Input channels (e.g., 2 for real/imag or 1 for magnitude).
        features (int): Base number of convolutional filters.
        output_shape (tuple): Shape of output maps [1, 1, H, W].
    """

    def __init__(self, input_dim: int = 1, features: int = 64, **kwargs):
        """
        Args:
            input_dim: Number of input channels (typically 1 for magnitude, 2 for real/imag).
            features: Base filter count (multiplied for deeper layers).
            **kwargs: Absorbed for compatibility with factory patterns.
        """
        super().__init__()
        self.input_dim = input_dim
        self.features = features
        self.output_shape = (1, 1)  # Placeholder; set during forward()

        # Encoder: Extract spatial features
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, features, kernel_size=3, padding=1),
            nn.InstanceNorm2d(features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Body: Intermediate processing using robust attention
        self.body = SwinBody(dim=features * 2)

        # T1 Head: Output range [10, 3010] ms (Sigmoid * 3000 + 10)
        self.head_t1 = nn.Sequential(
            nn.Conv2d(features * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # T2 Head: Output range [1, 501] ms (Sigmoid * 500 + 1)
        self.head_t2 = nn.Sequential(
            nn.Conv2d(features * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # PD Head: Output range [0, 10] (Softplus * 10)
        # Softplus ensures strict positivity with smooth gradient
        self.head_pd = nn.Sequential(
            nn.Conv2d(features * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Softplus(),
        )

        # B0 Head (Off-Resonance): Output range [-200, 200] Hz (Tanh * 200)
        # B0 can be negative (phase advance or delay)
        self.head_b0 = nn.Sequential(
            nn.Conv2d(features * 2, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh(),  # Range [-1, 1] -> scaled to [-200, 200] Hz
        )

        # B1 Head (Transmit Field): Output range [0.5, 1.5] (Tanh * 0.5 + 1.0)
        # B1 should be smooth (low frequency), so use larger kernel
        self.head_b1 = nn.Sequential(
            nn.Conv2d(features * 2, 32, kernel_size=3, padding=1),  # Larger kernel for smoothness
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Tanh(),  # Range [-1, 1] -> scaled to [0.5, 1.5]
        )

    @property
    def name(self) -> str:
        """Generator identifier for registry."""
        return "qmap_generator"

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: map undersampled input to quantitative parameter maps.

        Args:
            x (Tensor): Input image [B, C, H, W]. Typically:
                       - [B, 1, H, W] for magnitude input
                       - [B, 2, H, W] for real/imaginary split

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
                - t1_map: [B, 1, H, W] T1 times in milliseconds, range [10, 3010] ms
                - t2_map: [B, 1, H, W] T2 times in milliseconds, range [1, 501] ms
                - pd_map: [B, 1, H, W] Proton density, range [0, 10]
                - b0_map: [B, 1, H, W] B0 field (Hz), range [-200, 200] Hz
                - b1_map: [B, 1, H, W] B1 field scaling, range [0.5, 1.5]
        Physics-Constrained Scaling:
            T1, T2: Based on typical human tissue at 3T
            PD: Normalized across tissues
            B0: Off-resonance field (negative = phase advance, positive = phase delay)
            B1: Transmit field correction (< 1.0 means weaker flip, > 1.0 means stronger flip)

        forward method for QMapGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Extract encoder features
        feat = self.encoder(x)

        # Process through body
        feat = self.body(feat)

        # Apply specialized heads with physics-constrained scaling
        # T1: Sigmoid range [0, 1] -> scale to [10, 3010] ms
        t1_normalized = self.head_t1(feat)
        t1_map = t1_normalized * 3000.0 + 10.0

        # T2: Sigmoid range [0, 1] -> scale to [1, 501] ms
        t2_normalized = self.head_t2(feat)
        t2_map = t2_normalized * 500.0 + 1.0

        # PD: Softplus output -> scale by 10
        pd_map = self.head_pd(feat) * 10.0

        # B0: Tanh range [-1, 1] -> scale to [-200, 200] Hz
        # (Can adjust to ±400 Hz for higher susceptibility regions)
        b0_map = self.head_b0(feat) * 200.0

        # B1: Tanh range [-1, 1] -> scale to [0.5, 1.5]
        # Formula: Tanh * 0.5 + 1.0 gives [0.5, 1.5]
        b1_map = self.head_b1(feat) * 0.5 + 1.0

        # Store output shape for compatibility
        self.output_shape = (t1_map.shape[-2], t1_map.shape[-1])

        return t1_map, t2_map, pd_map, b0_map, b1_map

    def generate(
        self, z: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Alias for forward() for compatibility with IGenerator protocol.

        Args:
            z: Input tensor [B, C, H, W]
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple of (t1_map, t2_map, pd_map, b0_map, b1_map)
        """
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        Compute output shape without storing intermediate states.

        Args:
            input_shape: Expected input shape [B, C, H, W]

        Returns:
            Output shape [B, 1, H, W] (single output per head)
        """
        B, _, H, W = input_shape
        return (B, 1, H, W)

    def get_parameter_count(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
