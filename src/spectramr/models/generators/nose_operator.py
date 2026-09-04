r"""NOSE -- Neural-Operator Signal Equation.

Casts contrast formation as a discretization-invariant neural operator
:math:`\mathcal G_\theta:(\boldsymbol\xi(\cdot),\boldsymbol\varphi)\mapsto I(\cdot)`
mapping the parameter-map *function* :math:`\boldsymbol\xi` and the acquisition
vector :math:`\boldsymbol\varphi` to the image *function*. Realised as a Fourier
neural operator (reusing
:class:`spectramr.models.blocks.fno_block.SpectralConv2d`), so by discretization-
invariance one trained operator evaluates at any spatial resolution -- a model
trained on high-field maps renders coarse ultra-low-field contrasts without
retraining. The acquisition vector is broadcast as input channels, placing
contrast formation in the parametric-surrogate setting where operator learning
generalises across :math:`\boldsymbol\varphi`.

Trained with a data term plus the physics-anchoring term
(:class:`spectramr.models.losses.physics_anchor_loss.PhysicsAnchorLoss`); as the
anchor weight :math:`\beta\to\infty` the operator recovers analytic synthetic MRI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional

from spectramr.models.blocks.fno_block import SpectralConv2d
from spectramr.models.registry import register_model


@register_model(
    name="nose_operator",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
)
class NeuralOperatorSignalEquation(nn.Module):
    r"""Fourier neural operator from ``(parameter map, acquisition vector)`` to image.

    Args:
        param_channels: Tissue-parameter channels (3 = rho, T1, T2).
        acq_dim: Acquisition-vector dimension (broadcast as input channels).
        width: Operator lifting width.
        modes: Retained Fourier modes per spatial axis (the resolution-invariant knob).
        out_channels: Output image channels.
    """

    def __init__(
        self,
        param_channels: int = 3,
        acq_dim: int = 5,
        width: int = 16,
        modes: int = 8,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.acq_dim = int(acq_dim)
        self.lift = nn.Conv2d(param_channels + acq_dim, width, kernel_size=1)
        self.spectral1 = SpectralConv2d(width, width, modes, modes)
        self.local1 = nn.Conv2d(width, width, kernel_size=1)
        self.spectral2 = SpectralConv2d(width, width, modes, modes)
        self.local2 = nn.Conv2d(width, width, kernel_size=1)
        self.project = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, params: torch.Tensor, acquisition: torch.Tensor) -> torch.Tensor:
        """Render an image from the parameter map and acquisition vector.

        Args:
            params: ``[B, param_channels, H, W]`` tissue-parameter map.
            acquisition: ``[B, acq_dim]`` acquisition vector.

        Returns:
            Image ``[B, out_channels, H, W]`` at the input resolution.
        """
        b, _, h, w = params.shape
        acq_map = acquisition[:, : self.acq_dim, None, None].expand(b, self.acq_dim, h, w)
        x = self.lift(torch.cat([params, acq_map], dim=1))
        x = functional.gelu(self.spectral1(x) + self.local1(x))
        x = functional.gelu(self.spectral2(x) + self.local2(x))
        return self.project(x)


__all__ = ["NeuralOperatorSignalEquation"]
