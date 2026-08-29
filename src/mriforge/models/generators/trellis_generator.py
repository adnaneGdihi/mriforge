"""TRELLIS Generator - 3D Asset Generation
=======================================

Implementation of Microsoft TRELLIS for 3D asset generation from 2D images.
Adapted for MRIForge project following SOLID principles.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mriforge.models.blocks.adapters import (
    DecoderAdapter3D,
    LatentGridSpec,
    SpatialResizer3D,
)
from mriforge.models.interfaces.models import IGenerator
from mriforge.models.registry import register_model
from mriforge.models.representations.slat.renderers import (
    TrellisGaussianRenderer,
    TrellisMeshRenderer,
    VolumeRenderer,
)


class ReshapeTransform(nn.Module):
    """Helper module to reshape tensors for transformer processing."""

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ):
        """__init__.

        Args:
            input_shape (tuple[int, ...]): Description.
            output_shape (tuple[int, ...]): Description.
        """
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for ReshapeTransform.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x.view(x.shape[0], *self.output_shape)


@register_model(name="trellis", training_mode="reconstruction")
class TRELLISGenerator(IGenerator, nn.Module):
    """TRELLIS Generator for 3D asset generation from 2D images.

    This generator implements the TRELLIS architecture for converting
    2D images into 3D assets using multi-view reconstruction and
    radiance field modeling.

    Attributes:
        name (str): Model identifier
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        resolution (Tuple[int, int, int]): Target 3D resolution
        num_views (int): Number of multi-view images to generate
        backbone: Feature extraction backbone
        radiance_field: Radiance field decoder
        renderer: Volume renderer using differentiable ray marching (currently via renderer_decoder)

    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        resolution: tuple[int, int, int] = (64, 128, 128),
        coarse_resolution: tuple[int, int, int] | None = None,
        input_resolution: tuple[int, int] = (64, 64),
        patch_size: int = 8,
        stride: int = 8,
        num_views: int = 4,
        feature_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        num_stages: int = 3,  # Number of refinement stages
        renderer_type: str = "volume",  # "gaussian", "mesh", or "volume"
        num_gaussians: int = 8192,  # Number of Gaussian primitives for splatting
        **kwargs: Any,
    ):
        """Initialize TRELLIS generator.

        Args:
            in_channels: Number of input image channels
            out_channels: Number of output channels
            resolution: Target 3D resolution (D, H, W)
            coarse_resolution: Optional coarse resolution for latent grid
            input_resolution: Expected 2D input resolution (H, W).
                Must be divisible by stride.
            patch_size: Patch size for backbone ViT-like processing
            stride: Stride for patch embedding (must divide input_resolution
                dimensions)
            num_views: Number of multi-view images
            feature_dim: Feature dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            num_stages: Number of refinement stages for deep supervision

        Note:
            Resizing Policy: Input images are bilinearly interpolated to match
            input_resolution if they differ. The backbone requires
            input_resolution to be divisible by stride for proper patch
            processing.

        """
        super().__init__()

        self._name = "trellis"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.input_resolution = input_resolution
        default_coarse = LatentGridSpec(resolution).compute_latent_grid()
        self.coarse_resolution = coarse_resolution or default_coarse
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.num_stages = num_stages
        if renderer_type not in ("gaussian", "mesh", "volume"):
            raise ValueError(
                f"Unknown renderer_type {renderer_type!r}; "
                "expected one of 'gaussian', 'mesh', 'volume'"
            )
        self.renderer_type = renderer_type
        self.num_gaussians = num_gaussians

        # Feature extraction backbone (simplified ViT-like)
        self.backbone = self._build_backbone(
            in_channels,
            feature_dim,
            num_layers,
            num_heads,
            mlp_ratio,
            input_resolution,
            patch_size,
            stride,
        )

        # Multi-view generation head
        self.multi_view_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, num_views * feature_dim),
        )

        # Radiance field decoder
        self.radiance_field = self._build_radiance_field(feature_dim)

        # Renderer (Gaussian, Mesh, or Volume)
        self.renderer_decoder = self._build_renderer_decoder()
        self.slat_renderer = self._build_slat_renderer()

        # Latent decoder adapter
        self.decoder_adapter = DecoderAdapter3D(
            latent_dim=feature_dim // 2,
            out_channels=feature_dim // 2,
            coarse_resolution=self.coarse_resolution,
            target_resolution=self.coarse_resolution,
        )

        # Optional upsampler to final resolution
        self.output_resizer: SpatialResizer3D | None
        if self.coarse_resolution == self.resolution:
            self.output_resizer = None
        else:
            self.output_resizer = SpatialResizer3D(self.resolution)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv3d(feature_dim, out_channels, 1),
            nn.Tanh(),  # Ensure output in [-1, 1] range
        )

        # Learnable query for multi-view aggregation
        self._agg_query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        # Feedback projection for multi-stage refinement
        self._feedback_proj = nn.Linear(out_channels, feature_dim // 2)

        self.logger = logging.getLogger(__name__)

    def _build_backbone(
        self,
        in_channels: int,
        feature_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        input_resolution: tuple[int, int],
        patch_size: int,
        stride: int,
    ) -> nn.Module:
        """Build feature extraction backbone."""
        # Use configurable patches for flexible input resolutions
        height, width = input_resolution
        if height % stride != 0 or width % stride != 0:
            raise ValueError(
                "TRELLIS backbone expects input_resolution divisible by "
                f"stride={stride}; received {height}x{width}"
            )
        spatial_height = height // stride
        spatial_width = width // stride
        spatial_tokens = spatial_height * spatial_width

        return nn.Sequential(
            # Patch embedding
            nn.Conv2d(
                in_channels,
                feature_dim,
                kernel_size=patch_size,
                stride=stride,
            ),
            nn.LayerNorm([feature_dim, spatial_height, spatial_width]),
            # Reshape for transformer: [B, C, H, W] -> [B, H*W, C]
            ReshapeTransform(
                (feature_dim, spatial_height, spatial_width),
                (spatial_tokens, feature_dim),
            ),
            # Transformer blocks
            *[TransformerBlock(feature_dim, num_heads, mlp_ratio) for _ in range(num_layers)],
            # Reshape back for spatial operations: [B, H*W, C] -> [B, C, H, W]
            ReshapeTransform(
                (spatial_tokens, feature_dim),
                (feature_dim, spatial_height, spatial_width),
            ),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim),
        )

    def _build_radiance_field(self, feature_dim: int) -> nn.Module:
        """Build radiance field decoder."""
        return nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
        )

    def _build_renderer_decoder(self) -> nn.Module:
        """Build decoder head for the selected renderer type."""
        if self.renderer_type == "gaussian":
            return self._build_gaussian_decoder()
        elif self.renderer_type == "mesh":
            return self._build_mesh_decoder()
        else:  # "volume" fallback
            return self._build_volume_decoder()

    def _build_gaussian_decoder(self) -> nn.Module:
        """Build decoder for Gaussian splatting using Conv3d for volumetric compatibility.

        Uses 1x1x1 convolutions instead of Linear layers to handle 5D [B, C, D, H, W]
        outputs from DecoderAdapter3D. The final output is globally pooled + projected
        to (B, N, 14) Gaussian params: xyz(3) + scale(3) + rotation(4) + color(3) + opacity(1).
        """
        return nn.Sequential(
            nn.Conv3d(self.feature_dim // 2, self.feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.num_gaussians * 14),
        )

    def _build_mesh_decoder(self) -> nn.Module:
        """Build decoder for mesh vertices using Conv3d for volumetric compatibility."""
        return nn.Sequential(
            nn.Conv3d(self.feature_dim // 2, self.feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.num_gaussians * 3),  # vertex positions
        )

    def _build_volume_decoder(self) -> nn.Module:
        """Build decoder for volumetric output (original simplified renderer)."""
        return nn.Sequential(
            nn.Conv3d(self.feature_dim // 2, self.feature_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(self.feature_dim, self.feature_dim, 3, padding=1),
        )

    def _build_slat_renderer(self):
        """Build SLAT renderer for post-processing decoder output."""
        if self.renderer_type == "gaussian":
            return TrellisGaussianRenderer()
        elif self.renderer_type == "mesh":
            return TrellisMeshRenderer()
        else:
            return VolumeRenderer()

    @property
    def name(self) -> str:
        """Returns the model name."""
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TRELLIS generator.

        Args:
            x: Input 2D image tensor of shape (B, C, H, W)

        Returns:
            Generated 3D volume tensor of shape (B, C, D, H, W)

        """
        return self.forward_with_intermediates(x)[-1]  # Return final output only

    def forward_with_intermediates(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass with intermediate outputs for deep supervision.

        Args:
            x: Input 2D image tensor of shape (B, C, H, W)

        Returns:
            List of intermediate 3D volumes from each refinement stage,
            with the final output as the last element.

        """
        # Shape guards
        assert x.dim() == 4, f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D"
        assert x.shape[1] == self.in_channels, (
            f"Expected {self.in_channels} input channels, got {x.shape[1]}"
        )
        assert x.shape[2] >= 32 and x.shape[3] >= 32, f"Input spatial dims too small: {x.shape[2:]}"

        # Ensure input matches configured resolution for backbone compatibility
        if (x.shape[2], x.shape[3]) != self.input_resolution:
            x = F.interpolate(
                x,
                size=self.input_resolution,
                mode="bilinear",
                align_corners=False,
            )

        batch_size = x.shape[0]

        # Extract features from input image
        features = self.backbone(x)  # (B, feature_dim)

        # Generate multi-view features
        multi_view_features = self.multi_view_head(features)
        multi_view_features = multi_view_features.view(
            batch_size,
            self.num_views,
            self.feature_dim,
        )  # (B, num_views, feature_dim)

        # Aggregate multi-view features using cross-attention
        # Query: learnable aggregation token, Keys/Values: multi-view features

        # Expand query for batch
        query = self._agg_query.expand(batch_size, -1, -1)  # [B, 1, D]

        # Cross-attention: query attends to all views
        # Attention weights: softmax(Q @ K^T / sqrt(d))
        attn_scale = self.feature_dim**-0.5
        attn_weights = (
            torch.bmm(query, multi_view_features.transpose(1, 2)) * attn_scale
        )  # [B, 1, num_views]
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Weighted aggregation
        aggregated_features = torch.bmm(attn_weights, multi_view_features).squeeze(1)  # [B, D]

        # Initialize with base radiance features
        current_features = self.radiance_field(aggregated_features)
        # (B, feature_dim // 2)

        intermediate_outputs = []

        # Multi-stage refinement with deep supervision
        for stage in range(self.num_stages):
            # Decode latent features to coarse 3D grid
            coarse_volume = self.decoder_adapter(current_features)

            # Apply renderer decoder on the coarse grid
            rendered_volume = self.renderer_decoder(coarse_volume)

            if self.output_resizer is not None:
                rendered_volume = self.output_resizer(rendered_volume)

            # Project to output channels
            stage_output = self.output_proj(rendered_volume)

            # Store intermediate output for supervision
            intermediate_outputs.append(stage_output)

            # Prepare features for next stage (refinement)
            if stage < self.num_stages - 1:
                # Use current output features to refine for next stage
                # In practice, this would involve some feedback mechanism
                # For now, we'll use a simple refinement approach
                current_features = self._refine_features(current_features, stage_output)

        # Final shape validation
        final_output = intermediate_outputs[-1]
        expected_shape = (
            batch_size,
            self.out_channels,
            self.resolution[0],
            self.resolution[1],
            self.resolution[2],
        )
        assert final_output.shape == expected_shape, (
            f"Output shape {final_output.shape} doesn't match expected shape {expected_shape}"
        )

        return intermediate_outputs

    def _refine_features(
        self, current_features: torch.Tensor, stage_output: torch.Tensor
    ) -> torch.Tensor:
        """Refine features for the next stage based on current output.

        This implements a simple feedback mechanism where the current stage's
        output is used to refine the features for the next stage.

        Args:
            current_features: Current radiance features (B, feature_dim//2)
            stage_output: Output from current stage (B, C, D, H, W)

        Returns:
            Refined features for next stage
        """
        # Simple refinement: extract features from output and combine
        # with current features
        # Global average pool the stage output to get feedback features
        feedback_features = stage_output.mean(dim=[2, 3, 4])  # (B, C)

        # Project feedback to match feature dimension
        projected_feedback = self._feedback_proj(feedback_features)

        # Combine current features with feedback (residual connection)
        refined_features = current_features + projected_feedback

        return refined_features

    def generate(self, z: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Generate 3D assets from latent space.

        Args:
            z: Latent tensor (can be 2D image or latent vector)
            **kwargs: Additional generation parameters

        Returns:
            Generated 3D volume

        """
        # If z is a latent vector, we need to decode it first
        if z.dim() == 2:  # Latent vector
            batch_size = z.shape[0]
            latent_dim = z.shape[1]
            target_pixels = self.input_resolution[0] * self.input_resolution[1]

            if latent_dim < target_pixels:
                padding = target_pixels - latent_dim
                z = torch.cat(
                    (
                        z,
                        torch.zeros(
                            batch_size,
                            padding,
                            device=z.device,
                            dtype=z.dtype,
                        ),
                    ),
                    dim=1,
                )
            elif latent_dim > target_pixels:
                z = z[:, :target_pixels]

            z_reshaped = z.view(
                batch_size,
                1,
                self.input_resolution[0],
                self.input_resolution[1],
            )

            return self.forward(z_reshaped)
        if z.dim() == 4:  # 2D image
            return self.forward(z)
        raise ValueError(f"Unsupported latent dimension: {z.dim()}")

    def get_output_shape(
        self,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Returns the output shape for a given input shape.

        Args:
            input_shape: Input tensor shape (B, C, H, W)

        Returns:
            Output shape (B, C, D, H, W)

        """
        batch_size = input_shape[0]
        return (batch_size, self.out_channels, *self.resolution)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Simplified transformer block for TRELLIS backbone."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        """__init__.

        Args:
            dim (int): Description.
            num_heads (int): Description.
            mlp_ratio (float): Description.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for TransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x
