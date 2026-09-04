"""C_n-equivariant activation normalization.

Standard ActNorm uses one scale/bias per channel; that breaks
equivariance because cyclically permuting the group axis would permute
the (distinct) per-channel parameters. To stay equivariant the scale and
bias are *shared* across all ``group_order`` elements of each group
orbit: parameters are stored per *base* channel and broadcast across the
group axis. A cyclic shift of the group axis then commutes with the
affine transform.

The log-Jacobian-determinant is ``H * W * group_order * sum_{base} logs``
(each base scale acts on ``group_order`` channels).

Reference: J. Köhler et al., "Equivariant flows," ICML 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupActNorm2d(nn.Module):
    """Group-shared per-orbit affine normalization.

    Args:
        num_channels: Total channel count (divisible by ``group_order``).
        group_order: Order ``n`` of the cyclic group.
        eps: Numerical floor for data-dependent init.

    Raises:
        ValueError: If ``num_channels`` is not divisible by ``group_order``.
    """

    def __init__(self, num_channels: int, group_order: int = 8, eps: float = 1e-6) -> None:
        super().__init__()
        n = int(group_order)
        if num_channels % n != 0:
            raise ValueError(
                f"GroupActNorm2d requires channels divisible by group_order={n}, "
                f"got {num_channels}."
            )
        self.group_order = n
        self.base_channels = num_channels // n
        self.eps = float(eps)
        self.logs = nn.Parameter(torch.zeros(1, self.base_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.base_channels, 1, 1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        # Python-side memo of the ``initialized`` buffer. The buffer stays the
        # serialization SSOT (it must survive state_dict round-trips, or a
        # resumed model would re-run the data-dependent init and overwrite the
        # trained ``logs``/``bias``); this flag only spares the hot path the
        # ``.item()`` device sync that reading it costs on every forward.
        self._initialized = False

    def _broadcast(self, p: torch.Tensor) -> torch.Tensor:
        # [1, base, 1, 1] -> [1, base * n, 1, 1] (share across group axis).
        return p.repeat_interleave(self.group_order, dim=1)

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        b, _, h, w = x.shape
        # Reshape to expose the group orbit and average over it too, so the
        # init respects the parameter sharing.
        xr = x.view(b, self.base_channels, self.group_order, h, w)
        mean = xr.mean(dim=(0, 2, 3, 4), keepdim=False).view(1, self.base_channels, 1, 1)
        std = xr.std(dim=(0, 2, 3, 4), keepdim=False).view(1, self.base_channels, 1, 1)
        self.bias.data.copy_(-mean / (std + self.eps))
        self.logs.data.copy_(-torch.log(std + self.eps))
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._initialized and not reverse:
            # One device sync on the first forward of this process, none after.
            if not bool(self.initialized.item()):
                self._initialize(x)
            self._initialized = True

        b, _, h, w = x.shape
        logs = self._broadcast(self.logs)
        bias = self._broadcast(self.bias)
        log_det = self.logs.sum() * float(h * w) * self.group_order
        log_det = log_det.expand(b)

        if reverse:
            y = (x - bias) * torch.exp(-logs)
            return y, -log_det
        y = x * torch.exp(logs) + bias
        return y, log_det
