"""Guided Diffusion Generator
==========================

A generator designed for guided diffusion models, allowing for classifier guidance
or classifier-free guidance during the generation process.
"""

import torch
from torch import nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.reconstruction.unet import StandardUNetGenerator
from mriforge.models.registry import register_model


@register_model(name="guided_diffusion", training_mode="diffusion")
class GuidedDiffusionGenerator(nn.Module, IGenerator):
    """Guided Diffusion Generator.

    This model extends the standard U-Net to support guidance mechanisms.
    It typically accepts additional inputs like class labels or guidance scales.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
        num_classes: int = 0,
        guidance_scale: float = 1.0,
        time_dim: int = 256,
        **kwargs,
    ):
        """Initialize Guided Diffusion Generator.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Feature map sizes
            num_classes: Number of classes for conditional generation
            guidance_scale: Default guidance scale
            time_dim: Dimension for time embeddings
        """
        super().__init__()
        self.guidance_scale = guidance_scale
        self.num_classes = num_classes
        self.time_dim = time_dim

        # Multi-layer Perceptron to map classes into the same embedding space as time
        self.use_class_cond = num_classes > 0
        if self.use_class_cond:
            self.class_embed = nn.Embedding(num_classes, time_dim)
            self.class_mlp = nn.Sequential(
                nn.Linear(time_dim, time_dim * 4), nn.SiLU(), nn.Linear(time_dim * 4, time_dim)
            )

        # We configure the inner UNet to accept time embeddings
        self.generator = StandardUNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            time_dim=time_dim,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor
            time: Optional time steps for diffusion [N]
            class_labels: Optional class labels for conditioning
            guidance_scale: Optional guidance scale override
            **kwargs: Extra arguments

        Returns:
            Generated output

        forward method for GuidedDiffusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            class_labels (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            guidance_scale (float | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Proper context embedding incorporating both time and classes
        cond_kwargs = {}

        if time is not None:
            cond_kwargs["time"] = time

        if self.use_class_cond and class_labels is not None:
            c_emb = self.class_embed(class_labels)
            c_emb = self.class_mlp(c_emb)

            # The underlying ConfigurableUNet injects `contrast_emb` into the decoder.
            # While time_emb is used everywhere, providing class_labels as `contrast_emb`
            # aligns with conditional generation mechanisms where global properties
            # condition the generative upsampling path.
            cond_kwargs["contrast_emb"] = c_emb

        return self.generator(x, **cond_kwargs, **kwargs)

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x, **kwargs)

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
        count = self.generator.get_parameter_count()
        if self.use_class_cond:
            count += sum(p.numel() for p in self.class_embed.parameters())
        return count

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "guided_diffusion"
