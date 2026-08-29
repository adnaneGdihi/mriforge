"""C_n group-equivariant 2D convolution (regular representation).

Implements a cyclic-group (``C_n``) convolution whose feature maps carry
the regular representation of ``C_n``. A feature tensor is laid out as
``[B, C, H, W]`` with ``C = base_channels * group_order``; the trailing
``group_order`` block of channels indexes the group elements.

A group convolution is realised by sharing a single base kernel across
all ``group_order`` output orientations: for output orientation ``r`` the
kernel is the base kernel with its *input* group axis cyclically shifted
by ``r``. This guarantees that a cyclic shift of the input group axis
(i.e. the action of a group element) produces the same cyclic shift on
the output — the equivariance property
``conv(rho_g x) = rho_g conv(x)``.

Reference: T. S. Cohen and M. Welling, "Group equivariant convolutional
networks," ICML 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupConv2d(nn.Module):
    """Cyclic-group equivariant convolution.

    Args:
        in_channels: Total input channels (must be divisible by
            ``group_order``).
        out_channels: Total output channels (must be divisible by
            ``group_order``).
        kernel_size: Spatial kernel size.
        group_order: Order ``n`` of the cyclic group ``C_n``.
        padding: Spatial padding (defaults to ``kernel_size // 2``).

    Raises:
        ValueError: If channel counts are not divisible by ``group_order``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        group_order: int = 8,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        n = int(group_order)
        if in_channels % n != 0 or out_channels % n != 0:
            raise ValueError(
                f"GroupConv2d requires channels divisible by group_order={n}; "
                f"got in={in_channels}, out={out_channels}."
            )
        self.group_order = n
        self.in_base = in_channels // n
        self.out_base = out_channels // n
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2 if padding is None else int(padding)

        # Base kernel: [out_base, in_base, n, k, k]; the `n` axis is the
        # input group axis that gets cyclically rolled per output orientation.
        self.weight = nn.Parameter(
            torch.empty(self.out_base, self.in_base, n, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(self.out_base))

    def _expanded_weight(self) -> torch.Tensor:
        n = self.group_order
        # Build a [out_base * n, in_base * n, k, k] kernel by rolling the
        # input group axis for each of the n output orientations.
        oriented = []
        for r in range(n):
            rolled = torch.roll(self.weight, shifts=r, dims=2)
            # [out_base, in_base * n, k, k]
            oriented.append(
                rolled.reshape(self.out_base, self.in_base * n, self.kernel_size, self.kernel_size)
            )
        # Stack orientations along the output group axis -> [out_base*n, in_base*n, k, k].
        w = torch.stack(oriented, dim=1)  # [out_base, n, in_base*n, k, k]
        w = w.reshape(self.out_base * n, self.in_base * n, self.kernel_size, self.kernel_size)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._expanded_weight()
        bias = self.bias.repeat_interleave(self.group_order)
        return F.conv2d(x, weight, bias, padding=self.padding)
