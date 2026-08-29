"""Evidential UNet Generator.

Implements Pillar 9 of the ultra-low field synthesis paradigms. This generator
produces a 4-channel prediction grid representing the parameters of a
Normal-Inverse-Gamma (NIG) distribution:
    - gamma: predicted mean
    - nu: virtual observation count for mean estimation
    - alpha: virtual observation count for variance estimation
    - beta: sum of squared errors

Reference: Amini et al. 'Deep Evidential Regression'
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mriforge.models.reconstruction.unet import StandardUNetGenerator as StandardUNet
from mriforge.models.registry import register_model


class EvidentialMapping(nn.Module):
    """Maps raw network logits to valid NIG parameters.

    Transforms the 4-channel output into (gamma, nu, alpha, beta) while
    enforcing the strict mathematical constraints of the NIG distribution.
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        # We assume the generator outputs 4 channels
        if in_channels != 4:
            raise ValueError(f"Evidential mapping expects 4 incoming channels, got {in_channels}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Raw logits [B, 4, ...]
        Returns:
            Concatenated [gamma, nu, alpha, beta] with applied constraints

        forward method for EvidentialMapping.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        gamma = x[:, 0:1]  # Mean: Linear, no constraint

        # nu > 0 (virtual observations for the mean)
        nu = F.softplus(x[:, 1:2]) + 1e-4

        # alpha > 1 (virtual observations for the variance, ensures finite variance)
        alpha = F.softplus(x[:, 2:3]) + 1.0 + 1e-4

        # beta > 0 (sum of squared errors)
        beta = F.softplus(x[:, 3:4]) + 1e-4

        return torch.cat([gamma, nu, alpha, beta], dim=1)


@register_model(name="evidential_unet", training_mode="reconstruction")
class EvidentialUNet(nn.Module):
    """Evidential UNet for robust epistemic and aleatoric uncertainty quantification.

    Acts as a wrapper/extension over a standard 3D UNet, mapping the final layer
    to 4 distinct output channels parameterized as a Normal-Inverse-Gamma distribution.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,  # MUST be 4 for evidential parameters
        features: tuple[int, ...] = (32, 64, 128, 256),
        spatial_dims: int = 2,
        **kwargs: Any,
    ):
        super().__init__()

        if out_channels != 4:
            raise ValueError(
                f"EvidentialUNet must output exactly 4 channels for NIG parameters, "
                f"but out_channels={out_channels}."
            )

        self.spatial_dims = spatial_dims

        # Base feature extraction
        self.unet = StandardUNet(
            in_channels=in_channels,
            out_channels=max(features),  # Output heavy feature map
            features=features,
            **kwargs,
        )

        # Final prediction heads mapping latent features to the 4 parameters
        # [FIX] Use Conv2d/InstanceNorm2d for 2D MRI data (default); Conv3d for 3D volumes
        if spatial_dims == 3:
            Conv = nn.Conv3d
            Norm = nn.InstanceNorm3d
        else:
            Conv = nn.Conv2d
            Norm = nn.InstanceNorm2d

        self.head = nn.Sequential(
            Conv(max(features), 64, kernel_size=3, padding=1),
            Norm(64),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(64, 4, kernel_size=1),
        )

        self.evidential_mapping = EvidentialMapping(in_channels=4)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input volume [B, in_channels, D, H, W]
        Returns:
            Evidential parameters [B, 4, D, H, W]

        forward method for EvidentialUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        features = self.unet(x)
        logits = self.head(features)
        nig_params = self.evidential_mapping(logits)

        return nig_params
