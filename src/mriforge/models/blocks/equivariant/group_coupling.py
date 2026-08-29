"""C_n-equivariant affine coupling layer.

The coupling splits channels along the *base* (orbit) axis so that each
half still carries complete group orbits, then conditions the active half
on the passive half through a stack of :class:`GroupConv2d` layers. Because
the conditioning network is itself ``C_n``-equivariant and the split keeps
whole orbits intact, the coupling commutes with the group action.

The Jacobian is block-triangular, so ``log|det J| = sum log_s`` (zero for
additive coupling).

Reference: J. Köhler et al., "Equivariant flows," ICML 2020; T. S. Cohen
and M. Welling, "Group equivariant convolutional networks," ICML 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mriforge.models.blocks.equivariant.group_conv import GroupConv2d


class GroupCoupling(nn.Module):
    """Group-equivariant affine (or additive) coupling.

    Args:
        num_channels: Total channels (``base * group_order``); the base
            channel count must be even so the orbit split is balanced.
        group_order: Order ``n`` of the cyclic group.
        hidden_base: Base-channel width of the conditioning sub-network.
        coupling_type: ``"affine"`` or ``"additive"``.

    Raises:
        ValueError: On invalid divisibility or unknown ``coupling_type``.
    """

    def __init__(
        self,
        num_channels: int,
        group_order: int = 8,
        hidden_base: int = 32,
        coupling_type: str = "affine",
    ) -> None:
        super().__init__()
        n = int(group_order)
        if num_channels % n != 0:
            raise ValueError(
                f"GroupCoupling requires channels divisible by group_order={n}, got {num_channels}."
            )
        base = num_channels // n
        if base % 2 != 0:
            raise ValueError(f"GroupCoupling requires an even base-channel count, got {base}.")
        if coupling_type not in ("affine", "additive"):
            raise ValueError(
                f"Unknown coupling_type {coupling_type!r}; expected 'affine' or 'additive'."
            )
        self.group_order = n
        self.base = base
        self.split_base = base // 2
        self.coupling_type = coupling_type
        out_mult = 2 if coupling_type == "affine" else 1

        in_ch = self.split_base * n
        hid_ch = hidden_base * n
        out_ch = self.split_base * out_mult * n
        last = GroupConv2d(hid_ch, out_ch, kernel_size=3, group_order=n)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(
            GroupConv2d(in_ch, hid_ch, kernel_size=3, group_order=n),
            nn.ReLU(),
            last,
        )
        self._split_ch = self.split_base * n

    def _params(self, x_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x_a)
        if self.coupling_type == "affine":
            t, log_s = out.chunk(2, dim=1)
            log_s = torch.tanh(log_s)
            return log_s, t
        return torch.zeros_like(out), out

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        x_a, x_b = x[:, : self._split_ch], x[:, self._split_ch :]
        log_s, t = self._params(x_a)

        if reverse:
            y_b = (x_b - t) * torch.exp(-log_s)
            log_det = -log_s.flatten(1).sum(dim=1)
        else:
            y_b = x_b * torch.exp(log_s) + t
            log_det = log_s.flatten(1).sum(dim=1)

        y = torch.cat([x_a, y_b], dim=1)
        return y, log_det
