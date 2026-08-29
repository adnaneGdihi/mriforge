"""Bloch Cycle Network
===================

A physics-informed network that incorporates Bloch equation constraints
or cycle consistency for MRI reconstruction.
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.reconstruction.unet import AttentionType, BlockType, NormalizationType
from mriforge.models.reconstruction.unet import UNet as ConfigurableUNetGenerator
from mriforge.models.registry import register_model


@register_model(
    name="bloch_cycle",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
)
@register_model(
    name="bloch_cycle_network",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=False,
)
class BlochCycleNetwork(nn.Module, IGenerator):
    """Bloch Cycle Network for MRI Reconstruction.

    This model implements a physics-informed network where the forward pass
    predicts quantitative tissue maps (PD, T1, T2) and the backward pass
    (decoder) uses the analytical Bloch equations to reconstruct the signal.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        depth: int = 4,
        bilinear: bool = True,
        dropout: float = 0.0,
        normalization: str = "batch",
        activation: str = "relu",
        sequence_type: str = "SE",
        **kwargs,
    ):
        """Initialize Bloch Cycle Network.

        Args:
            in_channels: Number of input channels (e.g. k-space or image)
            out_channels: Number of output channels (target image)
            features: Feature map sizes
            depth: Network depth
            bilinear: Upsampling mode
            dropout: Dropout rate
            normalization: Normalization type
            activation: Activation function
            sequence_type: MRI sequence type ("SE" or "GRE")
        """
        super().__init__()

        # 1. Forward Generator (Encoder): Image -> Q-Maps (PD, T1, T2)
        # We enforce 3 output channels for the quantitative maps
        self.map_generator = ConfigurableUNetGenerator(
            in_channels=in_channels,
            out_channels=3,  # PD, T1, T2
            features=features,
            depth=depth,
            bilinear=bilinear,
            dropout=dropout,
            attention_type=AttentionType.CHANNEL,
            norm_type=NormalizationType.BATCH,
            block_type=BlockType.STANDARD,
            use_attention=True,
            use_residual=True,
            deep_supervision=False,  # We usually want just the final maps
        )

        # 2. Backward Generator (Decoder): Q-Maps -> Image
        # Analytical Differentiable Physics Simulator
        from mriforge.infrastructure.physics.bloch_simulation import (
            DifferentiableBlochSimulator,
        )

        self.bloch_simulator = DifferentiableBlochSimulator(sequence_type=sequence_type)

        # Default acquisition parameters (can be overridden in forward)
        # Spin Echo defaults
        self.default_tr = 3000.0  # ms
        self.default_te = 100.0  # ms

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W]
            kwargs:
                tr: Repetition time (ms)
                te: Echo time (ms)

        Returns:
            Reconstructed image (Signal) [B, 1, H, W]

        forward method for BlochCycleNetwork.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle NHWC -> NCHW if necessary
        needs_permute = x.ndim == 4 and x.shape[1] > 16 and x.shape[3] <= 16
        if needs_permute:
            x = x.permute(0, 3, 1, 2)

        # Handle complex input: extract magnitude to get real-valued tensor
        if torch.is_complex(x):
            x = x.abs()  # [B, C, H, W] magnitude

        # 1. Predict Quantitative Maps (PD, T1, T2)
        q_maps = self.map_generator(x)

        # Apply physical constraints (positivity)
        # T1, T2 must be positive. PD positive.
        q_maps = torch.abs(q_maps)

        # 2. Simulate Signal using Bloch Equations
        tr = kwargs.get("tr", self.default_tr)
        te = kwargs.get("te", self.default_te)

        signal = self.bloch_simulator(q_maps, TR=tr, TE=te)

        # Signal is [B, 1, H, W] (PD is channel 0, etc. handled in simulator)

        if needs_permute:
            signal = signal.permute(0, 2, 3, 1)

        return signal

    def forward_with_intermediates(self, x: torch.Tensor, **kwargs):
        """Forward pass enabling deep supervision or visualization."""
        # Similar logic but return maps
        if x.ndim == 4 and x.shape[1] > 16 and x.shape[3] <= 16:
            x = x.permute(0, 3, 1, 2)

        q_maps = torch.abs(self.map_generator(x))
        tr = kwargs.get("tr", self.default_tr)
        te = kwargs.get("te", self.default_te)
        signal = self.bloch_simulator(q_maps, TR=tr, TE=te)

        # Return q_maps as intermediates AND signal as final
        return signal, [q_maps, signal]

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        # Output shape matches input spatial dims, 1 channel
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        B, C, H, W = input_shape
        return (B, 1, H, W)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return self.map_generator.get_parameter_count()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "bloch_cycle_network"
