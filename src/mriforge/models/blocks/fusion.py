"""Fusion Blocks for MRI Reconstruction

This module provides various fusion strategies for combining multiple
reconstruction approaches in MRI super-resolution tasks.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .interfaces import IFusionStrategy


class AttentionFusionBlock(IFusionStrategy):
    """Attention-based fusion that learns to weight different inputs."""

    def __init__(
        self,
        channels: int,
        num_inputs: int = 2,
        reduction_ratio: int = 16,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_inputs (int): Description.
            reduction_ratio (int): Description.
        """
        super().__init__()
        self.num_inputs = num_inputs

        # Attention mechanism
        # Ensure at least 1 channel in the reduction path to avoid zero-sized conv
        reduced = max(1, channels // reduction_ratio)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels * num_inputs, 1),
            nn.Sigmoid(),
        )

        # Feature refinement
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse inputs using learned attention weights."""
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} inputs, got {len(inputs)}",
            )

        # Concatenate all inputs
        concat_features = torch.cat(inputs, dim=1)

        # Generate attention weights
        attention_weights = self.attention(concat_features)

        # Split weights for each input
        weights_split = torch.split(attention_weights, inputs[0].shape[1], dim=1)

        # Apply attention weights
        weighted_inputs = []
        for i, inp in enumerate(inputs):
            weighted = inp * weights_split[i]
            weighted_inputs.append(weighted)

        # Sum weighted inputs
        fused = sum(weighted_inputs)

        # Refine fused features
        output = self.refine(fused)

        return output


class FeatureFusionBlock(IFusionStrategy):
    """Feature-level fusion with channel attention and spatial attention."""

    def __init__(self, channels: int, num_inputs: int = 2):
        """__init__.

        Args:
            channels (int): Description.
            num_inputs (int): Description.
        """
        super().__init__()
        self.num_inputs = num_inputs

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * num_inputs, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels * num_inputs, 1),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid(),
        )

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse inputs using channel and spatial attention."""
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"Expected {self.num_inputs} inputs, got {len(inputs)}",
            )

        # Concatenate inputs
        concat_features = torch.cat(inputs, dim=1)

        # Channel attention
        channel_weights = self.channel_attention(concat_features)
        channel_weighted = concat_features * channel_weights

        # Split back to individual inputs
        split_features = torch.split(channel_weighted, inputs[0].shape[1], dim=1)

        # Apply spatial attention to each input
        spatially_weighted = []
        for inp in split_features:
            # Spatial attention per input
            spatial_weights = self.spatial_attention(inp)
            weighted = inp * spatial_weights
            spatially_weighted.append(weighted)

        # Sum spatially weighted features
        fused = sum(spatially_weighted)

        # Refine
        output = self.refine(fused)

        return output


class MultiScaleFusionBlock(IFusionStrategy):
    """Multi-scale fusion for combining features at different resolutions."""

    def __init__(self, channels: int, num_scales: int = 3):
        """__init__.

        Args:
            channels (int): Description.
            num_scales (int): Description.
        """
        super().__init__()
        self.num_scales = num_scales

        # Scale-specific processing
        self.scale_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_scales)]
        )

        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(channels, num_heads=8)

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_scales, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse multi-scale features."""
        if len(inputs) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} inputs, got {len(inputs)}")

        # Process each scale
        processed_scales = []
        for i, inp in enumerate(inputs):
            processed = self.scale_convs[i](inp)
            processed_scales.append(processed)

        # Cross-scale attention
        # Flatten spatial dimensions for attention
        batch_size, channels, h, w = processed_scales[0].shape
        flattened_scales = []
        for scale in processed_scales:
            flattened = scale.view(batch_size, channels, -1).permute(
                2,
                0,
                1,
            )  # (H*W, B, C)
            flattened_scales.append(flattened)

        # Apply cross-attention
        attended_scales = []
        for _, scale in enumerate(flattened_scales):
            attended, _ = self.cross_attention(scale, scale, scale)
            attended_scales.append(
                attended.permute(1, 2, 0).contiguous().view(batch_size, channels, h, w),
            )

        # Concatenate and fuse
        concat_features = torch.cat(attended_scales, dim=1)
        fused = self.fusion_conv(concat_features)

        return fused


class AdaptiveFusionBlock(IFusionStrategy):
    """Adaptive fusion that learns optimal combination weights."""

    def __init__(
        self,
        channels: int,
        num_inputs: int = 2,
        temperature: float = 1.0,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_inputs (int): Description.
            temperature (float): Description.
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.temperature = temperature

        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * num_inputs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_inputs),
            nn.Softmax(dim=1),
        )

        # Feature modulation
        self.modulation = nn.Conv2d(channels, channels, 1)

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse inputs with learned adaptive weights."""
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        # Concatenate inputs for weight prediction
        concat_features = torch.cat(inputs, dim=1)

        # Predict weights
        if weights is None:
            weights_tensor = self.weight_net(concat_features)  # (B, num_inputs)
        else:
            weights_tensor = torch.tensor(
                weights,
                device=inputs[0].device,
                dtype=inputs[0].dtype,
            ).unsqueeze(0)
            weights_tensor = weights_tensor.expand(inputs[0].shape[0], -1)

        # Apply weights to inputs
        weighted_inputs = []
        for i, inp in enumerate(inputs):
            weight = weights_tensor[:, i].view(-1, 1, 1, 1)
            weighted = inp * weight
            weighted_inputs.append(weighted)

        # Sum weighted inputs
        fused = sum(weighted_inputs)

        # Modulate features
        output = self.modulation(fused)

        return output


class ResidualFusionBlock(IFusionStrategy):
    """Residual fusion that preserves input information."""

    def __init__(self, channels: int, num_inputs: int = 2):
        """__init__.

        Args:
            channels (int): Description.
            num_inputs (int): Description.
        """
        super().__init__()
        self.num_inputs = num_inputs

        # Residual processing
        self.residual_conv = nn.Conv2d(channels, channels, 3, padding=1)

        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_inputs))

        # Output refinement
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse inputs with residual connections."""
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {len(inputs)}")

        # Use provided weights or learned weights
        if weights is None:
            fusion_weights = F.softmax(self.fusion_weights, dim=0)
        else:
            fusion_weights = torch.tensor(
                weights,
                device=inputs[0].device,
                dtype=inputs[0].dtype,
            )

        # Weighted sum
        fused = sum(w * inp for w, inp in zip(fusion_weights, inputs, strict=False))

        # Residual connection
        residual = self.residual_conv(fused)
        output = fused + residual

        # Refine
        output = self.refine(output)

        return output


class MultiModalFusionBlock(IFusionStrategy):
    """Fusion block for combining different reconstruction modalities."""

    def __init__(
        self,
        channels: int,
        num_modalities: int = 3,
        fusion_type: str = "attention",
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_modalities (int): Description.
            fusion_type (str): Description.
        """
        super().__init__()
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type

        if fusion_type == "attention":
            self.fusion_block = AttentionFusionBlock(channels, num_modalities)
        elif fusion_type == "feature":
            self.fusion_block = FeatureFusionBlock(channels, num_modalities)
        elif fusion_type == "adaptive":
            self.fusion_block = AdaptiveFusionBlock(channels, num_modalities)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Modality-specific processing
        self.modality_convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(num_modalities)],
        )

        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        weights: list[float] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Fuse different reconstruction modalities."""
        if len(inputs) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, got {len(inputs)}",
            )

        # Process each modality
        processed_modalities = []
        for i, inp in enumerate(inputs):
            processed = self.modality_convs[i](inp)
            processed_modalities.append(processed)

        # Fuse modalities
        fused = self.fusion_block(processed_modalities, weights)

        # Final refinement
        output = self.final_refine(fused)

        return output


def create_fusion_block(
    fusion_type: str,
    channels: int,
    num_inputs: int = 2,
    **kwargs: Any,
) -> nn.Module:
    """Factory function for creating fusion blocks."""
    if fusion_type.lower() == "attention":
        return AttentionFusionBlock(channels, num_inputs, **kwargs)
    if fusion_type.lower() == "feature":
        return FeatureFusionBlock(channels, num_inputs, **kwargs)
    if fusion_type.lower() == "multiscale":
        return MultiScaleFusionBlock(channels, num_inputs, **kwargs)
    if fusion_type.lower() == "adaptive":
        return AdaptiveFusionBlock(channels, num_inputs, **kwargs)
    if fusion_type.lower() == "residual":
        return ResidualFusionBlock(channels, num_inputs, **kwargs)
    if fusion_type.lower() == "multimodal":
        return MultiModalFusionBlock(channels, num_inputs, **kwargs)
    raise ValueError(f"Unknown fusion type: {fusion_type}")


def fuse_reconstructions(
    reconstructions: list[torch.Tensor],
    fusion_type: str = "attention",
    channels: int | None = None,
) -> torch.Tensor:
    """Convenience function for fusing multiple reconstructions."""
    if len(reconstructions) < 2:
        raise ValueError("Need at least 2 reconstructions to fuse")

    if channels is None:
        channels = reconstructions[0].shape[1]

    fusion_block = create_fusion_block(fusion_type, channels, len(reconstructions))
    fused = fusion_block(reconstructions)

    return fused
