"""Multi-Contrast Conditioning Generator for MRI Super-Resolution.

This module implements conditional generators that can leverage multiple MRI
contrasts (T1, T2, FLAIR) as auxiliary inputs for improved super-resolution
performance. The architecture includes learned fusion mechanisms and
cross-contrast feature integration.
"""

import logging
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


class ContrastFusionBlock(nn.Module):
    """Block for fusing features from multiple MRI contrasts."""

    def __init__(self, in_channels: int, out_channels: int, num_contrasts: int = 3):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_contrasts (int): Description.
        """
        super().__init__()
        self.num_contrasts = num_contrasts

        # Individual contrast processing
        self.contrast_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels // num_contrasts, 3, padding=1),
                    nn.BatchNorm2d(out_channels // num_contrasts),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_contrasts)
            ],
        )

        # Cross-contrast attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=8,
            batch_first=True,
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        hr_contrast: torch.Tensor,
        lr_contrasts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse high-resolution contrast with low-resolution auxiliary contrasts.

        Args:
            hr_contrast: High-res main contrast [B, C, H, W]
            lr_contrasts: Low-res auxiliary contrasts
            [B, num_contrasts, C, H, W]

        forward method for ContrastFusionBlock.

        Executes PyTorch tensor operations.

        Args:
            hr_contrast (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            lr_contrasts (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if lr_contrasts is None:
            # No auxiliary contrasts - return HR contrast features
            hr_encoded = self.contrast_encoders[0](hr_contrast)
            return self.fusion_net(hr_encoded)

        batch_size, _, height, width = hr_contrast.shape

        # Process each auxiliary contrast
        contrast_features = []
        for i in range(self.num_contrasts):
            contrast_feat = self.contrast_encoders[i](lr_contrasts[:, i])
            contrast_features.append(contrast_feat)

        # Concatenate all contrast features
        multi_contrast_feat = torch.cat(contrast_features, dim=1)
        # [B, out_channels, H, W]

        # Apply cross-contrast attention
        # Reshape for attention: [B, H*W, C]
        feat_flat = multi_contrast_feat.flatten(2).transpose(1, 2)
        attn_out, _ = self.cross_attention(feat_flat, feat_flat, feat_flat)
        attn_out = attn_out.transpose(1, 2).view(batch_size, -1, height, width)

        # Fuse with main contrast
        hr_encoded = self.contrast_encoders[0](hr_contrast)
        # Use first encoder for HR
        fused = torch.cat([hr_encoded, attn_out], dim=1)

        return self.fusion_net(fused)


class MultiContrastConditionalGenerator(nn.Module):
    """Conditional generator leveraging multiple MRI contrasts."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_aux_contrasts: int = 2,
        base_channels: int = 64,
        num_residual_blocks: int = 16,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_aux_contrasts (int): Description.
            base_channels (int): Description.
            num_residual_blocks (int): Description.
        """
        super().__init__()
        self.num_aux_contrasts = num_aux_contrasts

        # Initial convolution
        self.conv_first = nn.Conv2d(in_channels, base_channels, 9, padding=4)

        # Contrast fusion at multiple scales
        self.fusion_blocks = nn.ModuleList(
            [
                ContrastFusionBlock(base_channels, base_channels, num_aux_contrasts)
                for _ in range(3)  # Fusion at different scales
            ],
        )

        # Residual blocks with contrast conditioning
        self.residual_blocks = nn.ModuleList(
            [
                ConditionalResidualBlock(base_channels, num_aux_contrasts)
                for _ in range(num_residual_blocks)
            ],
        )

        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )

        # Output convolution
        self.conv_last = nn.Conv2d(base_channels, out_channels, 9, padding=4)

        # Skip connections for contrast fusion
        self.skip_connections = nn.ModuleList(
            [nn.Conv2d(base_channels, base_channels, 1) for _ in range(len(self.fusion_blocks))],
        )

    def forward(
        self,
        x: torch.Tensor,
        aux_contrasts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
        x: Input LR image [B, C, H, W]
        aux_contrasts: Auxiliary contrast images
        [B, num_contrasts, C, H, W]

        forward method for MultiContrastConditionalGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            aux_contrasts (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initial feature extraction
        feat = self.conv_first(x)

        # Multi-scale contrast fusion
        fusion_outputs = []
        for i, fusion_block in enumerate(self.fusion_blocks):
            if aux_contrasts is not None:
                # Downsample aux contrasts for current scale if needed
                scale_factor = 2**i
                aux_scaled = F.interpolate(
                    aux_contrasts,
                    scale_factor=1 / scale_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                fused = fusion_block(feat, aux_scaled)
            else:
                # No auxiliary contrasts - use identity mapping
                fused = self.skip_connections[i](feat)

            fusion_outputs.append(fused)
            feat = fused

        # Residual processing with conditioning
        for block in self.residual_blocks:
            if aux_contrasts is not None:
                feat = block(feat, aux_contrasts)
            else:
                feat = block(feat, None)

        # Upsampling
        feat = self.upsampling(feat)

        # Output
        output = self.conv_last(feat)
        return output


class ConditionalResidualBlock(nn.Module):
    """Residual block with contrast conditioning."""

    def __init__(self, channels: int, num_aux_contrasts: int):
        """__init__.

        Args:
            channels (int): Description.
            num_aux_contrasts (int): Description.
        """
        super().__init__()
        self.channels = channels
        self.num_aux_contrasts = num_aux_contrasts

        # Main residual path
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

        # Contrast conditioning
        if num_aux_contrasts > 0:
            self.condition_net = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(num_aux_contrasts, channels // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 16, channels, 1),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        aux_contrasts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional contrast conditioning.

        forward method for ConditionalResidualBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            aux_contrasts (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = x

        # Main residual path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Apply contrast conditioning if available
        if aux_contrasts is not None and hasattr(self, "condition_net"):
            condition = self.condition_net(aux_contrasts)
            out = out * condition

        return out + residual


@register_model(name="multi_contrast_generator", training_mode="reconstruction")
class MultiContrastGenerator(nn.Module, IGenerator):
    """Multi-contrast conditional generator implementing
    IGenerator interface.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_aux_contrasts: int = 2,
        base_channels: int = 64,
        num_residual_blocks: int = 16,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            num_aux_contrasts (int): Description.
            base_channels (int): Description.
            num_residual_blocks (int): Description.
        """
        super().__init__()
        self.generator = MultiContrastConditionalGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            num_aux_contrasts=num_aux_contrasts,
            base_channels=base_channels,
            num_residual_blocks=num_residual_blocks,
        )
        self._name = "multi_contrast_generator"

    @property
    def name(self) -> str:
        """Returns the model name."""
        return self._name

    def forward(
        self,
        x: torch.Tensor,
        aux_contrasts: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            aux_contrasts (Optional[torch.Tensor]): Description.
        Returns:
            torch.Tensor: Description.

        forward method for MultiContrastGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            aux_contrasts (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.generator(x, aux_contrasts)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved image from input image."""
        aux_contrasts = kwargs.get("aux_contrasts")
        return self.forward(z, aux_contrasts)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        batch_size, channels, height, width = input_shape
        # Assume 4x upsampling
        return (batch_size, channels, height * 4, width * 4)

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return model parameters."""
        return super().parameters()

    def to(self, device: torch.device) -> "MultiContrastGenerator":
        """Move model to device."""
        super().to(device)
        return self


def create_multi_contrast_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    num_aux_contrasts: int = 2,
    **kwargs,
) -> MultiContrastGenerator:
    """Factory function for multi-contrast generator."""
    return MultiContrastGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_aux_contrasts=num_aux_contrasts,
        **kwargs,
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the multi-contrast generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create generator
    generator = create_multi_contrast_generator(
        in_channels=1,
        out_channels=1,
        num_aux_contrasts=2,
    )
    generator.to(device)  # Move to device

    # Test input shapes
    batch_size = 2
    lr_size = 64
    hr_size = 256

    # LR input
    lr_input = torch.randn(batch_size, 1, lr_size, lr_size).to(device)

    # Auxiliary contrasts (T2, FLAIR)
    aux_contrasts = torch.randn(batch_size, 2, lr_size, lr_size).to(device)

    # Generate with conditioning
    with torch.no_grad():
        output = generator.generate(lr_input, aux_contrasts=aux_contrasts)

    logger.debug(f"Input shape: {lr_input.shape}")
    logger.debug(f"Aux contrasts shape: {aux_contrasts.shape}")
    logger.debug(f"Output shape: {output.shape}")
    logger.debug(f"Parameter count: {generator.get_parameter_count():,}")

    # Test without auxiliary contrasts
    output_no_aux = generator.generate(lr_input)
    logger.debug(f"Output without aux shape: {output_no_aux.shape}")
