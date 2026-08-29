"""HyperMamba UNet — MambaUNet with External SSM Parameter Injection.

Wraps the existing MambaUNet architecture to accept externally-generated
State-Space Model (SSM) parameters from the HyperMambaBridge. This enables
scan-specific motion correction where the Mamba blocks' behavior is
dynamically conditioned on the Virtual Fiducial's distortion pattern.

Architecture:
    [HyperMambaBridge] → SSMParameters → [HyperMambaUNet] → Reconstruction

Reference:
    Ha et al., "Hypernetworks," ICLR 2017.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from mriforge.models.generators.hyper_mamba_bridge import HyperMambaBridge
from mriforge.models.generators.mamba_unet import MambaUNet
from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


@register_model(
    name="hyper_mamba_unet",
    training_mode="reconstruction",
    spatial_dims=(2, 3),
    # VF / motion-meta strategies IFFT k-space → image internally before
    # this backbone runs, so the model genuinely accepts either on-disk
    # domain. Declaring both keeps the spatial-dims safety check active
    # while letting data_model_compatibility pass for kspace datasets.
    input_domain=("image", "kspace"),
    output_domain=("image", "kspace"),
    accepts_complex=False,
    requires_paired_data=True,
)
class HyperMambaUNet(nn.Module):
    """MambaUNet extended with external SSM weight injection.

    Composed of:
    1. A HyperMambaBridge that generates SSM params from corrupted fiducial
    2. A standard MambaUNet backbone for image reconstruction

    The bridge outputs are injected into the Mamba layers of the backbone
    via the ``ssm_params`` forward argument.

    Args:
        in_channels: Input channels for the UNet backbone.
        out_channels: Output channels for the UNet backbone.
        features: Feature dimensions per encoder level.
        mamba_config: Configuration for Mamba blocks.
        bridge_latent_dim: HyperMambaBridge latent dimension.
        bridge_d_state: SSM state dimension.
        bridge_d_model: SSM model dimension.
        fiducial_channels: Input channels for fiducial (2 = real/imag).
        time_embedding_dim: Time embedding dimension for diffusion conditioning.

    Example:
        >>> model = HyperMambaUNet(in_channels=2, out_channels=2)
        >>> x = torch.randn(4, 2, 256, 256)
        >>> fiducial = torch.randn(4, 2, 256, 256)
        >>> out = model(x, corrupted_fiducial=fiducial)
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        features: list[int] | None = None,
        mamba_config: dict | None = None,
        bridge_latent_dim: int = 128,
        bridge_d_state: int = 16,
        bridge_d_model: int = 256,
        fiducial_channels: int = 2,
        time_embedding_dim: int | None = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize HyperMambaUNet.

        Args:
            in_channels: Input channels for the backbone.
            out_channels: Output channels for the backbone.
            features: Feature dimensions per encoder level.
            mamba_config: Mamba block configuration dict.
            bridge_latent_dim: Latent dim for the hypernetwork.
            bridge_d_state: Mamba d_state.
            bridge_d_model: Mamba d_model (should match features[0] or bottleneck).
            fiducial_channels: Channel count for fiducial input.
            time_embedding_dim: Time embedding dim for diffusion.
            **kwargs: Additional kwargs passed to MambaUNet.
        """
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256]

        # Count Mamba layers in the UNet (encoder + decoder levels)
        num_mamba_layers = len(features) * 2  # encoder + decoder

        # 1. HyperMamba Bridge (generates SSM params from fiducial)
        self.bridge = HyperMambaBridge(
            in_channels=fiducial_channels,
            out_channels=out_channels,
            latent_dim=bridge_latent_dim,
            mamba_d_state=bridge_d_state,
            mamba_d_model=bridge_d_model,
            num_mamba_layers=num_mamba_layers,
        )

        # 2. MambaUNet backbone (the main reconstruction network)
        self.backbone = MambaUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            mamba_config=mamba_config,
            time_embedding_dim=time_embedding_dim,
            **kwargs,
        )

        logger.info(
            "[HyperMambaUNet] Initialized: in=%d, out=%d, features=%s, "
            "bridge_latent=%d, total_params=%d",
            in_channels,
            out_channels,
            features,
            bridge_latent_dim,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        corrupted_fiducial: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with optional fiducial conditioning.

        Args:
            x: Input tensor [B, C, H, W].
            timesteps: Diffusion timesteps [B] (optional).
            corrupted_fiducial: Distorted fiducial [B, 2, H, W] (optional).
                If None, runs the backbone without hyper-conditioning.
            **kwargs: Additional kwargs passed to backbone.

        Returns:
            Reconstructed output [B, out_channels, H, W].

        forward method for HyperMambaUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            corrupted_fiducial (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Generate SSM params from fiducial if provided
        if corrupted_fiducial is not None:
            ssm_params = self.bridge(corrupted_fiducial)
            kwargs["ssm_params"] = ssm_params

        # Forward through backbone
        return self.backbone(x, timesteps=timesteps, **kwargs)

    def freeze_backbone(self) -> None:
        """Freeze backbone weights for TTO (only bridge + theta are optimized)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("[HyperMambaUNet] Backbone frozen for TTO.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone weights for full training."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("[HyperMambaUNet] Backbone unfrozen.")


__all__ = ["HyperMambaUNet"]
