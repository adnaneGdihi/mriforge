"""UNet Generator Implementation
============================

Concrete implementation of IGenerator for UNet-based generators.
Follows SOLID principles with single responsibility and dependency injection.
"""

import torch
import torch.nn as nn

from spectramr.models.blocks.block_registry import create_block
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import AttentionType, UNet
from spectramr.models.registry import register_model


@register_model(name="unet", training_mode="reconstruction", spatial_dims=(2,))
class UNetGenerator(IGenerator, nn.Module):
    """UNet-based generator implementation following SOLID principles.

    Single Responsibility: Only handles UNet generation
    Open/Closed: Can be extended without modification
    Liskov Substitution: Fully substitutable for any IGenerator
    Interface Segregation: Only implements needed interfaces
    Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        bilinear: bool = False,
        output_activation: nn.Module | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            bilinear (bool): Description.
            output_activation (Optional[nn.Module]): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.output_activation = output_activation or nn.Identity()

        # Validate input parameters
        assert isinstance(in_channels, int) and in_channels > 0, (
            "in_channels must be a positive integer"
        )
        assert isinstance(out_channels, int) and out_channels > 0, (
            "out_channels must be a positive integer"
        )
        assert isinstance(bilinear, bool), "bilinear must be a boolean"

        self.use_complex_conv = kwargs.get("use_complex_conv", False)

        block_type = "unet_complex_double_conv" if self.use_complex_conv else "unet_double_conv"
        down_type = "unet_complex_down" if self.use_complex_conv else "unet_down"
        up_type = "unet_complex_up" if self.use_complex_conv else "unet_up"

        # Encoder
        self.inc = create_block(block_type, in_channels=in_channels, out_channels=64)
        self.down1 = create_block(down_type, in_channels=64, out_channels=128)
        self.down2 = create_block(down_type, in_channels=128, out_channels=256)
        self.down3 = create_block(down_type, in_channels=256, out_channels=512)
        factor = 2 if bilinear else 1
        self.down4 = create_block(down_type, in_channels=512, out_channels=1024 // factor)

        # Decoder
        self.up1 = create_block(
            up_type, in_channels=1024, out_channels=512 // factor, bilinear=bilinear
        )
        self.up2 = create_block(
            up_type, in_channels=512, out_channels=256 // factor, bilinear=bilinear
        )
        self.up3 = create_block(
            up_type, in_channels=256, out_channels=128 // factor, bilinear=bilinear
        )
        self.up4 = create_block(up_type, in_channels=128, out_channels=64, bilinear=bilinear)
        self.outc = create_block(
            "unet_out_conv",
            in_channels=64,
            out_channels=out_channels,
            activation=nn.Identity(),
        )

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "unet"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet.

        forward method for UNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return self.output_activation(logits)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for UNet, z is the input image."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate output shape for given input shape."""
        # UNet typically preserves spatial dimensions
        n, _, h, w = input_shape
        return (n, self.out_channels, h, w)

    def get_parameter_count(self) -> int:
        """Count total parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AttentionUNetGenerator(UNet):
    """UNet Generator with integrated Attention mechanism.

    Wraps the configurable UNet from spectramr.models.reconstruction.unet
    pre-configured for attention.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        attention_type: str = "spatial",
        **kwargs,
    ):
        # Map string attention type to Enum
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            attention_type (str): Description.
        """
        try:
            attn_enum = AttentionType(attention_type.lower())
        except ValueError as e:
            valid = [a.value for a in AttentionType]
            raise ValueError(
                f"Unknown attention_type {attention_type!r}; expected one of {valid}"
            ) from e

        # Filter kwargs to match UNetConfig fields to avoid TypeErrors
        # This is a robust way to handle "extra" kwargs passed by ModelFactory
        valid_config_keys = UNet.__expected_kwargs__
        config_kwargs = {k: v for k, v in kwargs.items() if k in valid_config_keys}

        # Override/Set specific defaults for AttentionUNet
        config_kwargs["in_channels"] = in_channels
        config_kwargs["out_channels"] = out_channels
        config_kwargs["use_attention"] = True
        config_kwargs["attention_type"] = attn_enum

        super().__init__(**config_kwargs)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "attention_unet"
