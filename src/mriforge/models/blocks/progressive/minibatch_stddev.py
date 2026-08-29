"""Minibatch standard-deviation layer for the progressive-GAN discriminator.

Implements the across-batch statistic of Karras *et al.* [Karras et al.,
ICLR 2018]. The layer appends a single feature map containing the mean
(over channels and space) of the per-pixel standard deviation computed
across the batch dimension. This hands the discriminator a global
statistic of activation variability, making intra-batch mode collapse
directly detectable.

.. math::

    s_{i,j} = \\frac{1}{C}\\sum_{c}
        \\sqrt{\\tfrac{1}{B}\\sum_{b}\\bigl(a_{b,c,i,j} - \\bar a_{c,i,j}\\bigr)^2 + \\epsilon},

then :math:`\\bar s = \\mathrm{mean}_{i,j}\\, s_{i,j}` is broadcast to a new
channel and concatenated to the input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MinibatchStdDev(nn.Module):
    """Append a constant minibatch-stddev feature channel.

    Args:
        group_size: Number of samples each statistic is computed over.
            The batch is split into groups of this size (the original
            ProGAN uses 4). If the batch is smaller than ``group_size``
            the whole batch forms a single group.
        epsilon: Numerical-stability constant added under the square root.
    """

    def __init__(self, group_size: int = 4, epsilon: float = 1e-8) -> None:
        super().__init__()
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.group_size = group_size
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate the minibatch-stddev statistic as one extra channel.

        Args:
            x: Activation tensor of shape ``[B, C, H, W]``.

        Returns:
            Tensor of shape ``[B, C + 1, H, W]``.
        """
        b, c, h, w = x.shape
        # Largest group size that evenly divides the batch (handles B < group_size).
        group_size = self.group_size
        if b % group_size != 0:
            group_size = b if b > 0 else 1
        n_groups = b // group_size

        # [G, group, C, H, W]
        y = x.reshape(n_groups, group_size, c, h, w)
        # variance over the group (sample) axis -> [G, C, H, W]
        y = y - y.mean(dim=1, keepdim=True)
        y = torch.sqrt((y * y).mean(dim=1) + self.epsilon)
        # mean over (C, H, W) -> [G, 1, 1, 1] then broadcast back to the batch.
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, h, w)
        return torch.cat([x, y], dim=1)

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}, epsilon={self.epsilon}"
