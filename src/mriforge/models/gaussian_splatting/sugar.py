"""Surface-Aligned Gaussian Splatting (SuGaR) Regularization.

Based on cluster.md Section 4.3 (Deep Dive):
"SuGaR adds a loss term that forces the distribution of Gaussians to align with the zero-level set of a Signed Distance Function (SDF)."
"""

from collections.abc import Callable

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from mriforge.models.registry import register_model


class SuGaRRegularizer(nn.Module):
    """Aligns 3D Gaussian Splats with an SDF surface."""

    def __init__(self, lambda_sdf: float = 1.0, lambda_normal: float = 0.1):
        """__init__.

        Args:
            lambda_sdf (float): Description.
            lambda_normal (float): Description.
        """
        super().__init__()
        self.lambda_sdf = lambda_sdf
        self.lambda_normal = lambda_normal

    def forward(
        self,
        means: Float[Tensor, "N 3"],
        scales: Float[Tensor, "N 3"],
        sdf_function: Callable[[Tensor], Tensor],
    ) -> Float[Tensor, ""]:
        """
        Args:
           means: Centers of the Gaussians.
           scales: Scales of Gaussians.
           sdf_function: A function f(x) -> dist that returns signed distance.

        Returns:
            loss: Scalar regularization loss.

        forward method for SuGaRRegularizer.

        Executes PyTorch tensor operations.

        Args:
            means (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            scales (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            sdf_function (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, '']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""

        # 1. Surface Alignment Loss
        # Hypothesis: Gaussian centers should lie on the surface (SDF=0)
        # or at least the density distribution implies the surface.
        # SuGaR paper enforces that density isosurface corresponds to SDF=0.
        # A simple proxy is minimizing abs(SDF(centers)).

        sdf_values = sdf_function(means)
        loss_surface = torch.abs(sdf_values).mean()

        # 2. Flatness / Normal Alignment (Optional)
        # Gaussians should be flat along the surface normal.
        # i.e., the smallest scale axis should align with SDF gradient.
        # This requires computing SDF gradient and comparing with Gaussian rotation.
        # We skip this for the basic implementation unless requested.

        loss = self.lambda_sdf * loss_surface

        return loss


@register_model(name="sugar_sdf", training_mode="experimental")
class SuGaRSDF(nn.Module):
    """A learnable SDF implicit function (MLP)."""

    def __init__(self, in_features=3, hidden_features=64, num_layers=4):
        """__init__.

        Args:
            in_features (Any): Description.
            hidden_features (Any): Description.
            num_layers (Any): Description.
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.Softplus(beta=100))
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.Softplus(beta=100))
        layers.append(nn.Linear(hidden_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: Float[Tensor, "B N 3"]) -> Float[Tensor, "B N 1"]:
        """forward.

        Args:
            coords (Float[Tensor, 'B N 3']): Description.
        Returns:
            Float[Tensor, 'B N 1']: Description.

        forward method for SuGaRSDF.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Float[Tensor, 'B N 1']: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(coords)
