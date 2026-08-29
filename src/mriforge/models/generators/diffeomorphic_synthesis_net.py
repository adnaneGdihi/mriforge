"""Diffeomorphic Synthesis Network.

Implements Pillar 10: Multi-Scale Jacobian Determinant Regularization.
Generates an image via a spatial deformation field, enforcing diffeomorphic constraints
on the tissue topology mapping between 64mT and 3T MRI contrasts.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.blocks.block_registry import create_block
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


@register_model(name="diffeomorphic_synthesis_net", training_mode="reconstruction")
class DiffeomorphicSynthesisNet(IGenerator, nn.Module):
    """Synthesis network that explicitly models diffeomorphic deformations.

    Outputs both the synthesized image and the underlying deformation field.
    The deformation field is extracted via `forward_with_intermediates` so it
    can be penalized by the HyperelasticJacobianLoss, ensuring topology
    preservation and no folding during synthesis.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        spatial_dims: int = 2,
        bilinear: bool = False,
        **kwargs: Any,
    ):
        """Initialize the Diffeomorphic Synthesis Network.

        Args:
            in_channels: Input channels (usually 2 for complex real/imag).
            out_channels: Output channels (usually 2 for complex real/imag).
            spatial_dims: 2 or 3 for 2D/3D deformations.
            bilinear: Use bilinear upsampling instead of transposed conv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims

        # To handle dynamic image sizes, we'll initialize the SpatialTransformer on-the-fly
        # or use grid_sample directly without needing predefined sizes.
        self.stn_mode = "bilinear"

        down_type = "unet_down"
        up_type = "unet_up"

        # Shared Encoder
        self.inc = create_block("unet_double_conv", in_channels=in_channels, out_channels=64)
        self.down1 = create_block(down_type, in_channels=64, out_channels=128)
        self.down2 = create_block(down_type, in_channels=128, out_channels=256)
        factor = 2 if bilinear else 1
        self.down3 = create_block(down_type, in_channels=256, out_channels=512 // factor)

        # Deformation Head (Decoder)
        self.def_up1 = create_block(
            up_type, in_channels=512, out_channels=256 // factor, bilinear=bilinear
        )
        self.def_up2 = create_block(
            up_type, in_channels=256, out_channels=128 // factor, bilinear=bilinear
        )
        self.def_up3 = create_block(up_type, in_channels=128, out_channels=64, bilinear=bilinear)

        # Output deformation field (dx, dy)
        self.def_outc = nn.Conv2d(64, spatial_dims, kernel_size=3, padding=1)

        # Initialize the deformation output to near-identity (small weights)
        nn.init.normal_(self.def_outc.weight, mean=0.0, std=1e-5)
        nn.init.constant_(self.def_outc.bias, 0.0)

        # Image Head (Decoder) - Refines features warped by the deformation field
        self.img_up1 = create_block(
            up_type, in_channels=512, out_channels=256 // factor, bilinear=bilinear
        )
        self.img_up2 = create_block(
            up_type, in_channels=256, out_channels=128 // factor, bilinear=bilinear
        )
        self.img_up3 = create_block(up_type, in_channels=128, out_channels=64, bilinear=bilinear)

        self.img_outc = create_block(
            "unet_out_conv",
            in_channels=64,
            out_channels=out_channels,
            activation=nn.Identity(),
        )

    @property
    def name(self) -> str:
        return "diffeomorphic_synthesis_net"

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard forward pass returning only the synthesized image.

        Args:
            x: Input tensor [B, C, H, W]

        forward method for DiffeomorphicSynthesisNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # When called directly (e.g. inference), we only care about the final image
        image, _ = self._forward_impl(x)
        return image

    def forward_with_intermediates(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        """Forward pass exposing the intermediate deformation field.

        Returns:
            Flat list ``[deformation_field, output_image]``.
            ``_generate_predictions`` in ``ReconstructionTrainingStrategy``
            takes ``[-1]`` as ``hr_fakes`` and passes the full list as
            ``intermediate_outputs`` to the loss computer.
        """
        image, deformation_field = self._forward_impl(x)
        return [deformation_field, image]

    def _warp_spatial(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Applies Spatial Transformer Network grid sampling without hardcoded sizes."""
        B, C, H, W = src.shape

        # Create base normalized meshgrid [-1, 1]
        vectors_y = torch.linspace(-1.0, 1.0, H, device=src.device)
        vectors_x = torch.linspace(-1.0, 1.0, W, device=src.device)
        grid_y, grid_x = torch.meshgrid(vectors_y, vectors_x, indexing="ij")

        # base_grid: [B, H, W, 2] where 2 channels = (x, y)
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Normalize flow from pixel space to [-1, 1] range
        # flow is [B, 2, H, W], we need strictly (x, y) order for grid_sample.
        # Assuming flow[:, 0] is dx, flow[:, 1] is dy.
        norm_flow_x = flow[:, 0, :, :] * (2.0 / (W - 1))
        norm_flow_y = flow[:, 1, :, :] * (2.0 / (H - 1))
        norm_flow = torch.stack([norm_flow_x, norm_flow_y], dim=-1)

        new_locs = base_grid + norm_flow

        # grid_sample expects input: [B, C, H, W], grid: [B, H, W, 2]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.stn_mode)

    def _forward_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Deformation Branch
        d = self.def_up1(x4, x3)
        d = self.def_up2(d, x2)
        d = self.def_up3(d, x1)
        deformation_field = self.def_outc(d)  # [B, 2, H, W]

        # Image Synthesis Branch
        # Warp the deepest feature representation and intermediate features
        wx4 = self._warp_spatial(
            x4,
            F.interpolate(
                deformation_field, size=x4.shape[2:], mode="bilinear", align_corners=True
            ),
        )
        wx3 = self._warp_spatial(
            x3,
            F.interpolate(
                deformation_field, size=x3.shape[2:], mode="bilinear", align_corners=True
            ),
        )
        wx2 = self._warp_spatial(
            x2,
            F.interpolate(
                deformation_field, size=x2.shape[2:], mode="bilinear", align_corners=True
            ),
        )
        wx1 = self._warp_spatial(x1, deformation_field)

        i = self.img_up1(wx4, wx3)
        i = self.img_up2(i, wx2)
        i = self.img_up3(i, wx1)

        # Warp original input features as local shortcut
        wx = self._warp_spatial(x, deformation_field)

        # The out convolution mixes warped context with refined hierarchical features
        image = self.img_outc(i) + wx

        return image, deformation_field

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(z)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        n, _, h, w = input_shape
        return (n, self.out_channels, h, w)

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["DiffeomorphicSynthesisNet"]
