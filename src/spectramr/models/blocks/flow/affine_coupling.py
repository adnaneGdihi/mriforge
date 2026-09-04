"""Affine / additive coupling layer for normalizing flows.

Channels are split as ``x = (x_a, x_b)``. The first half passes through
unchanged; the second half is transformed by a neural network
conditioned on the first:

    y_a = x_a
    y_b = x_b * exp(s_theta(x_a)) + t_theta(x_a)   (affine)
    y_b = x_b + t_theta(x_a)                        (additive)

The Jacobian is block-triangular with an identity top-left block, so
``log|det J| = sum s_theta(x_a)`` (zero for additive coupling).

Reference: D. P. Kingma and P. Dhariwal, "Glow: Generative flow with
invertible 1x1 convolutions," NeurIPS 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AffineCoupling(nn.Module):
    """Channel-wise affine (or additive) coupling.

    Args:
        num_channels: Total channel count ``C`` (must be even so the
            split is balanced).
        hidden_channels: Width of the conditioning sub-network.
        coupling_type: ``"affine"`` or ``"additive"``.

    Raises:
        ValueError: If ``num_channels`` is odd or ``coupling_type`` is
            not one of the two supported options (no silent fallback).
    """

    def __init__(
        self,
        num_channels: int,
        hidden_channels: int = 512,
        coupling_type: str = "affine",
    ) -> None:
        super().__init__()
        c = int(num_channels)
        if c % 2 != 0:
            raise ValueError(f"AffineCoupling requires an even channel count, got {c}.")
        if coupling_type not in ("affine", "additive"):
            raise ValueError(
                f"Unknown coupling_type {coupling_type!r}; expected 'affine' or 'additive'."
            )
        self.coupling_type = coupling_type
        self.split_size = c // 2
        out_mult = 2 if coupling_type == "affine" else 1

        conv1 = nn.Conv2d(self.split_size, hidden_channels, 3, padding=1)
        conv2 = nn.Conv2d(hidden_channels, hidden_channels, 1)
        conv3 = nn.Conv2d(hidden_channels, self.split_size * out_mult, 3, padding=1)
        # Zero-init the last layer so the layer starts as identity.
        nn.init.zeros_(conv3.weight)
        nn.init.zeros_(conv3.bias)
        self.net = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU(), conv3)

    def _params(self, x_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x_a)
        if self.coupling_type == "affine":
            t, log_s = out.chunk(2, dim=1)
            # Stabilise the scale (Glow uses a sigmoid-gated form).
            log_s = torch.tanh(log_s)
            return log_s, t
        zero = torch.zeros_like(out)
        return zero, out

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling.

        Args:
            x: ``[B, C, H, W]`` input.
            reverse: If ``True`` invert the transform.

        Returns:
            ``(y, log_det)`` with per-sample ``[B]`` log-determinant.
        """
        x_a, x_b = x[:, : self.split_size], x[:, self.split_size :]
        log_s, t = self._params(x_a)

        if reverse:
            y_b = (x_b - t) * torch.exp(-log_s)
            log_det = -log_s.flatten(1).sum(dim=1)
        else:
            y_b = x_b * torch.exp(log_s) + t
            log_det = log_s.flatten(1).sum(dim=1)

        y = torch.cat([x_a, y_b], dim=1)
        return y, log_det
