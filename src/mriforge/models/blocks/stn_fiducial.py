r"""Spatial Transformer Network for Virtual Fiducial Alignment.

Regresses a 6-DoF rigid affine matrix :math:`\theta` from a fiducial
coordinate channel, then applies differentiable grid sampling to lock
the reconstruction to the physical marker geometry:

.. math::

    \theta = \text{LocalizationNet}(x_{\text{fiducial}})
    \quad\in\mathbb{R}^{2\times3}

.. math::

    \text{grid} = F.\text{affine\_grid}(\theta, \text{size})

.. math::

    y = F.\text{grid\_sample}(\text{features}, \text{grid})

The localization network uses a lightweight CNN backbone followed by
two FC layers, outputting a 2×3 affine matrix initialized to the
identity transform (no initial warping).

Reference:
    Jaderberg et al., "Spatial Transformer Networks", *NeurIPS* (2015).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class STNFiducial(nn.Module):
    """Spatial Transformer for Virtual Fiducial registration.

    A lightweight localization network regresses an affine transform
    from fiducial marker coordinates, then applies differentiable
    grid sampling to spatially align features to the marker frame.

    Args:
        in_channels: Number of input channels (features + fiducial channel).
        fiducial_channels: Number of fiducial coordinate channels
            (default 1 for a single marker heatmap, 3 for x/y/z coords).
        feature_channels: Number of feature channels to transform.
            If ``None``, all channels except fiducial are features.
        localization_hidden: Hidden dimension in the localization CNN.
    """

    def __init__(
        self,
        in_channels: int = 3,
        fiducial_channels: int = 1,
        feature_channels: int | None = None,
        localization_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.fiducial_channels = fiducial_channels
        self.feature_channels = feature_channels or (in_channels - fiducial_channels)

        # Localization network: extract affine transform from fiducial channel
        self.localization_net = nn.Sequential(
            nn.Conv2d(fiducial_channels, localization_hidden, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(localization_hidden, localization_hidden * 2, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )

        # Regression head: output 6 parameters for 2×3 affine
        self.fc_loc = nn.Sequential(
            nn.Linear(localization_hidden * 2 * 16, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6),
        )

        # Initialize to identity transform
        self._init_identity()

    def _init_identity(self) -> None:
        """Initialize the regression head to output the identity affine."""
        # Last linear layer: bias = [1, 0, 0, 0, 1, 0] (identity)
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def get_affine_matrix(self, fiducial: Tensor) -> Tensor:
        """Regress the affine matrix from fiducial channels.

        Args:
            fiducial: ``(B, fiducial_channels, H, W)`` marker heatmap or coords.

        Returns:
            ``(B, 2, 3)`` affine transformation matrix.
        """
        xs = self.localization_net(fiducial)
        xs = xs.flatten(1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(
        self,
        x: Tensor,
        fiducial: Tensor | None = None,
    ) -> Tensor:
        """Apply spatial transformer to features.

        Args:
            x: ``(B, C, H, W)`` input features. If ``fiducial`` is None,
                the last ``fiducial_channels`` channels are used as the
                fiducial input.
            fiducial: Optional explicit fiducial input ``(B, Fc, H, W)``.
                If provided, ``x`` is treated as pure features.

        Returns:
            Spatially transformed features ``(B, C_feat, H, W)``.
        """
        if fiducial is None:
            # Split features and fiducial from input
            Fc = self.fiducial_channels
            fiducial = x[:, -Fc:]
            features = x[:, :-Fc]
        else:
            features = x

        # Regress affine from fiducial
        theta = self.get_affine_matrix(fiducial)

        # Generate sampling grid
        grid = F.affine_grid(theta, features.size(), align_corners=False)

        # Differentiable grid sampling
        output = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return output


class STNFiducial3D(nn.Module):
    """3D Spatial Transformer for volumetric fiducial registration.

    Extends STNFiducial to 3D volumes with a 3×4 affine matrix
    (12 parameters for full 3D rigid + scaling).

    Args:
        in_channels: Number of input channels.
        fiducial_channels: Number of fiducial coordinate channels.
        localization_hidden: Hidden dimension in localization CNN.
    """

    def __init__(
        self,
        in_channels: int = 3,
        fiducial_channels: int = 1,
        localization_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.fiducial_channels = fiducial_channels

        self.localization_net = nn.Sequential(
            nn.Conv3d(fiducial_channels, localization_hidden, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(localization_hidden, localization_hidden * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2),
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(localization_hidden * 2 * 8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 12),  # 3×4 affine
        )

        # Identity init
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
        )

    def forward(self, x: Tensor, fiducial: Tensor | None = None) -> Tensor:
        """Apply 3D spatial transform.

        Args:
            x: ``(B, C, D, H, W)`` volumetric features.
            fiducial: ``(B, Fc, D, H, W)`` or splits from x.

        Returns:
            Transformed ``(B, C_feat, D, H, W)``.
        """
        if fiducial is None:
            Fc = self.fiducial_channels
            fiducial = x[:, -Fc:]
            features = x[:, :-Fc]
        else:
            features = x

        xs = self.localization_net(fiducial)
        xs = xs.flatten(1)
        theta = self.fc_loc(xs).view(-1, 3, 4)

        grid = F.affine_grid(theta, features.size(), align_corners=False)
        output = F.grid_sample(
            features,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return output


__all__ = ["STNFiducial", "STNFiducial3D"]
