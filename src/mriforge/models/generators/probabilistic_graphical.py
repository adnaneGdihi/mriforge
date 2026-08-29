"""Probabilistic Graphical Model Generator for structured MRI reconstruction.

Uses a factor-graph style message-passing layer to capture spatial structure
in k-space / image domain through belief propagation.

Reference:
    Yedidia et al., "Understanding Belief Propagation and its Generalizations",
    IJCAI 2001.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model


class _MessagePassingLayer(nn.Module):
    """Single round of sum-product / max-product message passing on a grid."""

    def __init__(self, ch: int, num_factors: int = 8) -> None:
        super().__init__()
        self.num_factors = num_factors
        # Pairwise potentials (learned)
        self.pairwise = nn.Conv2d(ch, ch, 3, padding=1, groups=min(ch, num_factors))
        # Unary update
        self.unary = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 1),
        )
        self.gate = nn.Sequential(nn.Conv2d(ch * 2, ch, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method for _MessagePassingLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        msg = self.pairwise(x)
        unary = self.unary(x)
        gate = self.gate(torch.cat([msg, unary], dim=1))
        return x + gate * msg


@register_model(
    name="probabilistic_graphical",
    training_mode="reconstruction",
    # Measured, not inferred from the name (#1106). The whole stack is
    # ``Conv2d``/``GroupNorm``/``SiLU``: a 5-D input raises "Expected 3D or 4D
    # input to conv2d", so rank 2 is a contract.
    spatial_dims=(2,),
    # The module docstring hedges -- "spatial structure in k-space / image
    # domain" -- but the model itself is domain-agnostic (no FFT anywhere), so
    # the question is what it is FED, and both committed arms answer image:
    # experiment_105 declares training.input_domain/output_domain: image over
    # ``coils.processing_mode: rss_image`` (IFFT inside the dataset pipeline),
    # and the validated sibling declares output_domain: image. Same resolution as
    # gnn_coil_fusion in #1084 -- the arms were asserting what the model refused
    # to say about itself.
    input_domain="image",
    output_domain="image",
    # Measured: complex input raises "Input type (c10::complex<float>) and bias
    # type (float) should be the same" at the first conv. Real/imag-interleaved
    # channels are the supported route (both arms use in_channels 1 or 2).
    accepts_complex=False,
    requires_paired_data=True,
)
class ProbabilisticGraphicalGenerator(IGenerator, nn.Module):
    """PGM-based generator with iterative belief-propagation refinement."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        feature_dim: int = kwargs.get("feature_dim", 64)
        num_factors: int = kwargs.get("num_factors", 8)
        inference_steps: int = kwargs.get("inference_steps", 5)

        self.intro = nn.Conv2d(in_channels, feature_dim, 3, padding=1)
        self.bp_layers = nn.ModuleList(
            [_MessagePassingLayer(feature_dim, num_factors) for _ in range(inference_steps)]
        )
        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(32, feature_dim), feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, out_channels, 3, padding=1),
        )

    @property
    def name(self) -> str:
        return "probabilistic_graphical"

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """forward method for ProbabilisticGraphicalGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        h = self.intro(x)
        for bp in self.bp_layers:
            h = bp(h)
        return self.out_conv(h)

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self.out_channels, *input_shape[2:])

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
