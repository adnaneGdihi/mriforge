"""Cascaded Diffusion Generator
============================

A generator designed for cascaded diffusion models, typically involving
multiple stages of resolution or refinement.
"""

import torch
from torch import nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.reconstruction.unet import StandardUNetGenerator
from spectramr.models.registry import register_model


@register_model(name="cascaded_diffusion", training_mode="diffusion")
class CascadedDiffusionGenerator(nn.Module, IGenerator):
    """Cascaded Diffusion Generator.

    This model wraps a standard U-Net but is designed to be used in a cascaded
    pipeline. It can handle conditioning on low-resolution inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        low_res_conditioning: bool = False,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (tuple[int, ...]): Description.
            low_res_conditioning (bool): Description.
        """
        super().__init__()
        self.low_res_conditioning = low_res_conditioning

        # If conditioning on low-res, we might concatenate it to input
        # or use it as a separate condition. For simplicity, we assume concatenation
        # if low_res_conditioning is True.
        effective_in_channels = in_channels * 2 if low_res_conditioning else in_channels

        self.generator = StandardUNetGenerator(
            in_channels=effective_in_channels,
            out_channels=out_channels,
            features=features,
            **kwargs,
        )

    def forward(
        self, x: torch.Tensor, low_res: torch.Tensor = None, *args, **kwargs
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            low_res (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for CascadedDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            low_res (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.low_res_conditioning and low_res is not None:
            # Upsample low_res to match x if needed
            if low_res.shape[-2:] != x.shape[-2:]:
                low_res = nn.functional.interpolate(
                    low_res, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, low_res], dim=1)

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
        return "cascaded_diffusion"
