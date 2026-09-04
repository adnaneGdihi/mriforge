"""Invertible 1x1 convolution with LU parameterisation.

A 1x1 convolution applies a per-pixel linear map ``z = W x`` with
``W in R^{C x C}``. The naive log-determinant is ``O(C^3)``; the LU
parameterisation ``W = P L (U + diag(s))`` with a fixed permutation
``P``, unit-lower-triangular ``L``, strictly-upper-triangular ``U`` and
learnable diagonal ``s`` collapses the cost to ``O(C)`` because
``det P = +/-1``, ``det L = 1`` and ``det(U + diag(s)) = prod_c s_c``.
The spatial factor ``H * W`` arises because ``W`` is applied
independently at each spatial location.

Reference: D. P. Kingma and P. Dhariwal, "Glow: Generative flow with
invertible 1x1 convolutions," NeurIPS 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertibleConv1x1LU(nn.Module):
    """Invertible 1x1 conv parameterised through its LU decomposition.

    Args:
        num_channels: Channel count ``C`` of the ``[B, C, H, W]`` input.
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        c = int(num_channels)
        self.num_channels = c

        # Seed with a random rotation, then take its LU factorisation so
        # the initial map is volume-preserving and well-conditioned.
        w_init = torch.linalg.qr(torch.randn(c, c))[0]
        p, lower, upper = torch.linalg.lu(w_init)

        # Fixed permutation (not learned) — keep as a buffer.
        self.register_buffer("P", p)
        # Sign + log-magnitude of the U diagonal are the learnable scale.
        s = torch.diagonal(upper).clone()
        self.register_buffer("sign_s", torch.sign(s))
        self.log_s = nn.Parameter(torch.log(torch.abs(s) + 1e-6))

        # Lower (unit diagonal) and strictly-upper parts.
        self.lower = nn.Parameter(lower)
        upper_no_diag = upper - torch.diag(torch.diagonal(upper))
        self.upper = nn.Parameter(upper_no_diag)

        # Static masks for re-assembling L and U on each forward.
        self.register_buffer("l_mask", torch.tril(torch.ones(c, c), diagonal=-1))
        self.register_buffer("eye", torch.eye(c))

    def _assemble_w(self) -> torch.Tensor:
        lower = self.lower * self.l_mask + self.eye
        upper = self.upper * self.l_mask.transpose(0, 1)
        diag = self.sign_s * torch.exp(self.log_s)
        w = self.P @ lower @ (upper + torch.diag(diag))
        return w

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the 1x1 conv.

        Args:
            x: ``[B, C, H, W]`` input.
            reverse: If ``True`` apply ``W^{-1}``.

        Returns:
            ``(y, log_det)`` with per-sample ``[B]`` log-determinant.
        """
        b, _, h, w = x.shape
        weight = self._assemble_w()
        log_det = self.log_s.sum() * float(h * w)
        log_det = log_det.expand(b)

        if reverse:
            weight = torch.inverse(weight)
            log_det = -log_det

        y = F.conv2d(x, weight.unsqueeze(-1).unsqueeze(-1))
        return y, log_det
