"""Octave Convolution block.

Implements the octave convolution of Chen *et al.* (ICCV 2019, "Drop an
octave: reducing spatial redundancy in convolutional neural networks with
octave convolution").

Feature maps are decomposed into a high-frequency component
``X^H in R^{(1-alpha) C x H x W}`` and a low-frequency component
``X^L in R^{alpha C x H/2 x W/2}`` with ``alpha in [0, 1]`` the *octave
ratio*. A single octave convolution computes four information flows
simultaneously::

    Y^H = f(X^H; W_hh) + Up(f(X^L; W_lh))
    Y^L = f(Pool(X^H); W_hl) + f(X^L; W_ll)

with ``Pool`` a 2x2 average pool and ``Up`` a 2x bilinear upsample.

The MRI interpretation: the low-frequency stream carries low-``k``
structural content and the high-frequency stream carries high-``k``
detail, so the octave decomposition acts as a *learned* k-space band
split.

Canonical home for this reusable block is ``models/blocks/`` — it is
NEVER inlined inside a generator (see CLAUDE.md, canonical-homes rule).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["OctaveConv2d"]


def _split_channels(channels: int, alpha: float) -> tuple[int, int]:
    """Split ``channels`` into (high, low) counts for octave ``alpha``.

    The high-frequency stream gets ``ceil((1-alpha) * C)`` channels and
    the low-frequency stream gets ``int(alpha * C)`` channels. This
    matches the reference skeleton and guarantees ``high + low == C``
    only when ``alpha in {0, 1}``; for intermediate ``alpha`` the rounding
    follows the reference (ceil for high, floor for low).
    """
    low = int(alpha * channels)
    high = int(math.ceil((1.0 - alpha) * channels))
    return high, low


class OctaveConv2d(nn.Module):
    """Octave convolution operating on a (high, low) feature-map pair.

    The module holds up to four sub-convolutions corresponding to the
    four octave information flows: ``hh`` (high->high), ``hl``
    (high->low), ``lh`` (low->high), ``ll`` (low->low). Pathways whose
    in/out channel count is zero (e.g. when ``alpha_in == 0`` there is no
    low-frequency input) are set to ``None`` and skipped — there is no
    silent fallback to a different operator.

    Args:
        in_channels: Total input channel count (high + low).
        out_channels: Total output channel count (high + low).
        kernel_size: Convolution kernel size (square).
        alpha_in: Octave ratio of the *input* feature map in ``[0, 1]``.
        alpha_out: Octave ratio of the *output* feature map in ``[0, 1]``.
        stride: Convolution stride (applied within each pathway).
        bias: Whether the sub-convolutions carry a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        alpha_in: float = 0.5,
        alpha_out: float = 0.5,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha_in <= 1.0:
            raise ValueError(f"alpha_in must lie in [0, 1], got {alpha_in}")
        if not 0.0 <= alpha_out <= 1.0:
            raise ValueError(f"alpha_out must lie in [0, 1], got {alpha_out}")

        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.stride = stride

        in_h, in_l = _split_channels(in_channels, alpha_in)
        out_h, out_l = _split_channels(out_channels, alpha_out)
        self.in_h, self.in_l = in_h, in_l
        self.out_h, self.out_l = out_h, out_l

        pad = kernel_size // 2

        # Each pathway exists only when both its in- and out-channel
        # counts are positive. Storing None for absent pathways keeps the
        # forward branches explicit (no silent fallback).
        self.conv_hh = (
            nn.Conv2d(in_h, out_h, kernel_size, stride, pad, bias=bias)
            if in_h > 0 and out_h > 0
            else None
        )
        self.conv_hl = (
            nn.Conv2d(in_h, out_l, kernel_size, stride, pad, bias=bias)
            if in_h > 0 and out_l > 0
            else None
        )
        self.conv_lh = (
            nn.Conv2d(in_l, out_h, kernel_size, stride, pad, bias=bias)
            if in_l > 0 and out_h > 0
            else None
        )
        self.conv_ll = (
            nn.Conv2d(in_l, out_l, kernel_size, stride, pad, bias=bias)
            if in_l > 0 and out_l > 0
            else None
        )

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Run the four octave pathways.

        Args:
            x: Either a single high-frequency tensor (when the input has
                no low-frequency stream, ``alpha_in == 0``) or a
                ``(x_high, x_low)`` tuple. ``x_low`` may be ``None``.

        Returns:
            A ``(y_high, y_low)`` tuple. Either element may be ``None``
            when the corresponding output stream has zero channels.
        """
        if isinstance(x, (tuple, list)):
            x_h, x_l = x[0], x[1]
        else:
            x_h, x_l = x, None

        y_h: torch.Tensor | None = None
        y_l: torch.Tensor | None = None

        # --- contributions into the HIGH output stream ---------------
        if self.conv_hh is not None and x_h is not None:
            y_h = self.conv_hh(x_h)
        if self.conv_lh is not None and x_l is not None:
            # low -> high requires a 2x upsample to match high resolution
            lh = self.conv_lh(x_l)
            lh = F.interpolate(lh, scale_factor=2, mode="bilinear", align_corners=False)
            y_h = lh if y_h is None else y_h + lh

        # --- contributions into the LOW output stream ----------------
        if self.conv_ll is not None and x_l is not None:
            y_l = self.conv_ll(x_l)
        if self.conv_hl is not None and x_h is not None:
            # high -> low requires a 2x average pool to drop resolution
            hl = self.conv_hl(F.avg_pool2d(x_h, kernel_size=2))
            y_l = hl if y_l is None else y_l + hl

        return y_h, y_l
