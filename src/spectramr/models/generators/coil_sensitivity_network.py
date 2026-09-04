"""Coil Sensitivity Network
========================

A specialized network for estimating coil sensitivity maps in MRI.
This network typically takes a combined image or multi-coil data and estimates
the complex sensitivity profiles for each coil.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import AttentionType, BlockType, NormalizationType
from spectramr.models.reconstruction.unet import UNet as ConfigurableUNetGenerator
from spectramr.models.registry import register_model


@register_model(name="coil_sensitivity", training_mode="reconstruction")
class CoilSensitivityNetwork(nn.Module, IGenerator):
    """Coil Sensitivity Estimation Network.

    This model uses a U-Net architecture to estimate coil sensitivity maps
    from input MRI data. It is designed to handle complex-valued data
    represented as separate channels.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Default to complex input (real/imag)
        out_channels: int = 2,  # Default to complex output (real/imag)
        features: tuple[int, ...] = (64, 128, 256, 512),
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        normalization: str = "instance",  # Instance norm is common in MRI
        activation: str = "leaky",
        **kwargs,
    ):
        """Initialize Coil Sensitivity Network.

        Args:
            in_channels: Number of input channels (e.g., 2 for complex image)
            out_channels: Number of output channels (e.g., 2 * num_coils)
            features: Feature map sizes for each level
            depth: Number of downsampling/upsampling levels
            bilinear: Whether to use bilinear upsampling
            dropout: Dropout probability
            normalization: Normalization type
            activation: Activation function
        """
        super().__init__()
        self.generator = ConfigurableUNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            depth=depth,
            bilinear=bilinear,
            dropout=dropout,
            attention_type=AttentionType.NONE,
            norm_type=NormalizationType.INSTANCE,
            block_type=BlockType.STANDARD,  # Residual blocks often help
            use_attention=False,
            use_residual=True,
            deep_supervision=False,
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor [N, C, H, W]

        Returns:
            Estimated sensitivity maps [N, C_out, H, W]

        forward method for CoilSensitivityNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure input is normalized if needed, though usually handled by data loader
        return self.generator(x)

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return self.generator.get_output_shape(input_shape)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.generator.get_parameter_count()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "coil_sensitivity_network"
