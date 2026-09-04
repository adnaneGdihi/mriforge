"""Pixelwise feature normalisation for progressive-growing GANs.

Implements the local response normalisation variant of Karras *et al.*
[Karras et al., ICLR 2018] used in the generator of the progressive GAN.

Each activation vector at spatial location :math:`(i, j)` is scaled to
unit length across the channel axis:

.. math::

    b_{c,i,j} = \\frac{a_{c,i,j}}{\\sqrt{\\tfrac{1}{C}\\sum_{c'} a_{c',i,j}^2 + \\epsilon}}.

This prevents the generator activations from blowing up in either
direction during the competitive adversarial dynamics, without
introducing learnable parameters (unlike batch/instance norm).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PixelNorm(nn.Module):
    """Pixelwise feature-vector normalisation (parameter-free).

    Args:
        epsilon: Numerical-stability constant added under the square root.
    """

    def __init__(self, epsilon: float = 1e-8) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"PixelNorm epsilon must be positive, got {epsilon}")
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the per-pixel feature vector to (approximately) unit norm.

        Args:
            x: Activation tensor of shape ``[B, C, H, W]``.

        Returns:
            Tensor of the same shape with each pixel's channel vector
            divided by its RMS magnitude.
        """
        # mean over the channel axis (dim=1), keep dims for broadcasting.
        rms = torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.epsilon)
        return x * rms

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}"
