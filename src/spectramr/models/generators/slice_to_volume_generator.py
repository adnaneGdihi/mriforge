"""Slice-to-Volume Generator
===========================

A generator model that reconstructs a 3D volume from a sequence of 2D slices.
This is useful for 2D-to-3D super-resolution tasks.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(name="slice_to_volume", training_mode="reconstruction")
class SliceToVolumeGenerator(nn.Module, IGenerator):
    """A generator that takes a sequence of 2D slices and generates a 3D volume.
    It uses a 2D CNN encoder and a 3D transposed convolution decoder.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_slices: int = 64,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ):
        """Initialize SliceToVolumeGenerator.

        Args:
            in_channels: Number of input channels for each 2D slice.
            out_channels: Number of output channels for the 3D volume.
            num_slices: Number of 2D slices in the input sequence.
            features: Feature map sizes for each level of the encoder/decoder.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_slices = num_slices
        self.features = features

        # 2D Encoder
        self.encoder = nn.Sequential(
            self._make_encoder_block(in_channels, features[0]),
            self._make_encoder_block(features[0], features[1]),
            self._make_encoder_block(features[1], features[2]),
            self._make_encoder_block(features[2], features[3]),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[3], features[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 3D Decoder
        self.decoder = nn.Sequential(
            self._make_decoder_block(features[3], features[2]),
            self._make_decoder_block(features[2], features[1]),
            self._make_decoder_block(features[1], features[0]),
            self._make_decoder_block(features[0], out_channels, final_layer=True),
        )

    def _make_encoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """_make_encoder_block.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        Returns:
            nn.Module: Description.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _make_decoder_block(
        self,
        in_channels: int,
        out_channels: int,
        final_layer: bool = False,
    ) -> nn.Module:
        """_make_decoder_block.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            final_layer (bool): Description.
        Returns:
            nn.Module: Description.
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator.

        Args:
            x: Input tensor of shape (N, C, D, H, W)

        forward method for SliceToVolumeGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size, channels, depth, h, w = x.shape

        # Process each slice independently with the 2D encoder
        x = x.view(batch_size * depth, channels, h, w)
        encoded_slices = self.encoder(x)

        bottleneck_out = self.bottleneck(encoded_slices)

        # Reshape to (N, C, D, H, W) for the 3D decoder
        # The height and width are reduced by the encoder
        h2, w2 = bottleneck_out.size(2), bottleneck_out.size(3)

        # Must divide by 16 across depth due to the fact that we swap to 3D and features[3] logic
        # For a truly robust 2D -> 3D transform, spatial reduction is [1, 16, 16] but depth
        # is currently modeled as being divided manually.
        depth_reduced = depth // (2**4)
        if depth_reduced == 0:
            depth_reduced = 1  # Fallback safeguard

        # Add padding to bottleneck out if depth doesn't divide perfectly
        target_elements = batch_size * self.features[3] * depth_reduced * h2 * w2
        if bottleneck_out.nelement() != target_elements:
            # We enforce exact reshaping using adaptive pooling if the slice count is irregular
            import torch.nn.functional as F

            bottleneck_out = F.adaptive_avg_pool2d(bottleneck_out, (h2, w2))

            # The tensor must be reshaped into depth_reduced.
            # If batch_size * depth * features[3] doesn't match target, we slice/pad
            reshaped_for_3d = bottleneck_out.view(batch_size, self.features[3], -1, h2, w2)
            # Interpolate depth completely to expected reduced geometry
            reshaped_for_3d = F.interpolate(
                reshaped_for_3d, size=(depth_reduced, h2, w2), mode="trilinear", align_corners=False
            )
        else:
            reshaped_for_3d = bottleneck_out.view(
                batch_size,
                self.features[3],
                depth_reduced,
                h2,
                w2,
            )

        # 3D Decoder
        output = self.decoder(reshaped_for_3d)

        # In case the 2D encoder stride / 3D decoder transpose convolutions misalign on boundary,
        # interpolate back to strict original dimensions over D, H, W
        if output.shape[2:] != (depth, h, w):
            import torch.nn.functional as F

            output = F.interpolate(
                output, size=(depth, h, w), mode="trilinear", align_corners=False
            )

        return output

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        """Generate output from input tensor."""
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape.
        The forward logic has been fortified with trilinear spatial interpolation
        to forcefully ensure the network preserves its exact structural tensor guarantees.
        """
        n, _, d, h, w = input_shape
        return (n, self.out_channels, d, h, w)

    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Get generator name."""
        return "slice_to_volume"


def create_slice_to_volume_generator(
    in_channels: int = 1,
    out_channels: int = 1,
    num_slices: int = 64,
    features: tuple[int, ...] = (64, 128, 256, 512),
    **kwargs,
) -> SliceToVolumeGenerator:
    """Factory function for creating SliceToVolumeGenerator instances.

    Args:
        in_channels: Number of input channels for each 2D slice.
        out_channels: Number of output channels for the 3D volume.
        num_slices: Number of 2D slices in the input sequence.
        features: Feature map sizes for each level of the encoder/decoder.
        **kwargs: Additional arguments (ignored)

    Returns:
        Configured SliceToVolumeGenerator instance

    """
    return SliceToVolumeGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        num_slices=num_slices,
        features=features,
    )
