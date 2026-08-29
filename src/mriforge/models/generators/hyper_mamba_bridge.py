"""HyperMamba Bridge Network.

A Convolutional Hypernetwork that generates scan-specific Mamba SSM parameters
by analyzing the distortion pattern of a corrupted Virtual Fiducial. The bridge
converts physical motion information into dynamic neural network weights.

Architecture:
    Corrupted Fiducial → CNN Encoder → Latent Physics Descriptor →
    Linear Heads → (B_proj, C_proj, Δ_proj) for Mamba SSM

Reference:
    Ha et al., "Hypernetworks," ICLR 2017.
    Gu & Dao, "Mamba: Linear-Time Sequence Modeling," 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from mriforge.models.registry import register_model

logger = logging.getLogger(__name__)


@dataclass
class SSMParameters:
    """Container for generated State-Space Model parameters.

    Attributes:
        B_proj: Input projection matrix [B, d_model * d_state].
        C_proj: Output projection matrix [B, d_model * d_state].
        Delta_proj: Discretization step sizes [B, d_model].
    """

    B_proj: torch.Tensor
    C_proj: torch.Tensor
    Delta_proj: torch.Tensor


@register_model(name="hyper_mamba_bridge", training_mode="reconstruction")
class HyperMambaBridge(nn.Module):
    """Hypernetwork that generates Mamba SSM weights from a corrupted fiducial.

    The bridge network digests the purely physical distortion of the Virtual
    Fiducial (Real/Imag channels) and outputs the state-space matrices
    (B, C, Δ) that dynamically program the main Mamba reconstruction network.

    Instead of learning a generic deblurring filter, this enables
    scan-specific motion inversion: "I see a 2mm Y-axis translation at t=1.5s
    in the fiducial → generate SSM params that invert exactly that motion."

    Args:
        in_channels: Input channels (2 for real/imag fiducial).
        latent_dim: Dimension of the latent physics descriptor.
        mamba_d_state: Mamba state dimension (d_state in SSM).
        mamba_d_model: Mamba model dimension (d_model in SSM).
        num_mamba_layers: Number of Mamba layers to generate params for.

    Example:
        >>> bridge = HyperMambaBridge(in_channels=2, mamba_d_model=256)
        >>> corrupted_fiducial = torch.randn(4, 2, 256, 256)
        >>> ssm_params = bridge(corrupted_fiducial)
        >>> ssm_params.B_proj.shape  # [4, 256 * 16]
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        latent_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_model: int = 256,
        num_mamba_layers: int = 4,
        **kwargs: object,
    ) -> None:
        """Initialize HyperMambaBridge.

        Args:
            in_channels: Fiducial input channels (real/imag = 2).
            out_channels: Unused, kept for registry interface compatibility.
            latent_dim: Latent physics descriptor dimension.
            mamba_d_state: SSM state dimension.
            mamba_d_model: SSM model dimension.
            num_mamba_layers: Number of Mamba layers to parameterize.
            **kwargs: Additional kwargs (ignored, for registry compatibility).
        """
        super().__init__()
        self.mamba_d_state = mamba_d_state
        self.mamba_d_model = mamba_d_model
        self.num_mamba_layers = num_mamba_layers
        self.latent_dim = latent_dim

        # CNN encoder: digests the physical distortion of the fiducial
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(16, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.GroupNorm(32, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

        # Per-layer SSM parameter generators
        # Each Mamba layer gets its own B, C, Δ projections
        self.gen_B = nn.ModuleList(
            [nn.Linear(latent_dim, mamba_d_model * mamba_d_state) for _ in range(num_mamba_layers)]
        )
        self.gen_C = nn.ModuleList(
            [nn.Linear(latent_dim, mamba_d_model * mamba_d_state) for _ in range(num_mamba_layers)]
        )
        self.gen_Delta = nn.ModuleList(
            [nn.Linear(latent_dim, mamba_d_model) for _ in range(num_mamba_layers)]
        )

        # Initialize Delta heads with small positive bias (Δ must be > 0)
        for head in self.gen_Delta:
            nn.init.constant_(head.bias, 0.5)

        logger.info(
            "[HyperMambaBridge] Initialized: latent_dim=%d, d_state=%d, "
            "d_model=%d, num_layers=%d, total_params=%d",
            latent_dim,
            mamba_d_state,
            mamba_d_model,
            num_mamba_layers,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, corrupted_fiducial: torch.Tensor) -> list[SSMParameters]:
        """Generate per-layer SSM parameters from corrupted fiducial.

        Args:
            corrupted_fiducial: Distorted fiducial [B, 2, H, W] (real/imag).

        Returns:
            List of SSMParameters, one per Mamba layer.

        forward method for HyperMambaBridge.

        Executes PyTorch tensor operations.

        Args:
            corrupted_fiducial (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            list[SSMParameters]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Convert complex tensor to real form [B, C*2, H, W]
        if torch.is_complex(corrupted_fiducial):
            B_size, C_size, H, W = corrupted_fiducial.shape
            # view_as_real adds a feature dim of size 2 at the end: [B, C, H, W, 2]
            corrupted_fiducial = torch.view_as_real(corrupted_fiducial)
            corrupted_fiducial = corrupted_fiducial.permute(0, 1, 4, 2, 3).reshape(
                B_size, C_size * 2, H, W
            )

        # Encode the physical distortion pattern
        latent_physics = self.hyper_encoder(corrupted_fiducial)

        # Generate SSM matrices for each Mamba layer
        ssm_params_list = []
        for i in range(self.num_mamba_layers):
            B_proj = self.gen_B[i](latent_physics)
            C_proj = self.gen_C[i](latent_physics)
            # Delta must be strictly positive (discretization step)
            Delta_proj = torch.exp(self.gen_Delta[i](latent_physics))

            ssm_params_list.append(
                SSMParameters(
                    B_proj=B_proj,
                    C_proj=C_proj,
                    Delta_proj=Delta_proj,
                )
            )

        return ssm_params_list


__all__ = ["HyperMambaBridge", "SSMParameters"]
