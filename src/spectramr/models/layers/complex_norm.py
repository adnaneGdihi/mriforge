"""Phase- and spectrum-preserving normalization for complex k-space features.

Standard ``BatchNorm``/``GroupNorm`` normalize per-channel statistics computed
over the spatial grid, which flattens the 1/f energy distribution of k-space
(the DC bin carries orders of magnitude more energy than the outer lines). That
is why the pure-k-space ``complex_unet`` backbone left its normalization slots as
``nn.Identity`` — and, with nothing controlling inter-layer activation scale, the
network's magnitude drifts over training into the experiment_11
measurement-independent collapse (see ``docs/experiment_11_kspace_cold_diffusion``
and pitfall #20).

:class:`ComplexRMSNorm` fills that gap without touching the spectrum: it divides
by a single per-sample RMS scalar (so every channel and every frequency bin is
scaled identically — relative magnitudes, hence the spectrum shape, are exactly
preserved) and applies a learnable per-channel real gain (phase-preserving,
since a real scalar times a complex value keeps its argument).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ComplexRMSNorm(nn.Module):
    """Global RMS normalization for interleaved complex feature maps.

    Operates on the interleaved complex layout produced by
    :class:`~spectramr.models.layers.complex_conv.ComplexConv2d`, i.e.
    ``[B, 2C, H, W]`` ordered ``[r0, i0, r1, i1, ...]``.

    The normalizer is one positive scalar per sample,
    ``s = sqrt(mean_over(C,H,W) |z|^2)``, applied to real and imaginary parts
    alike. Because the same scalar multiplies every element, both the phase and
    the relative magnitude spectrum are preserved; only the overall scale is
    pinned, which is exactly the inter-layer scale control the k-space backbone
    lacks. A learnable per-channel real ``gain`` (init 1.0) restores expressivity.

    Args:
        complex_channels: number of complex channels ``C`` (half the real
            channel count ``2C``).
        eps: numerical floor added to the mean power before the reciprocal sqrt.
    """

    def __init__(self, complex_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.complex_channels = int(complex_channels)
        self.eps = float(eps)
        self.gain = nn.Parameter(torch.ones(self.complex_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = 2 * self.complex_channels
        if x.shape[1] != expected:
            raise ValueError(
                f"ComplexRMSNorm expects interleaved complex channels "
                f"[B, {expected}, H, W] for complex_channels={self.complex_channels}, "
                f"got channel dim {x.shape[1]}."
            )
        real = x[:, 0::2]
        imag = x[:, 1::2]
        # One RMS scalar per sample over (C, H, W): scale control that leaves the
        # relative spectrum untouched (unlike per-channel BN/GN).
        power = real * real + imag * imag
        inv = torch.rsqrt(power.mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        g = self.gain.view(1, -1, 1, 1)
        real = real * inv * g
        imag = imag * inv * g
        return torch.stack([real, imag], dim=2).flatten(1, 2)

    def extra_repr(self) -> str:
        return f"complex_channels={self.complex_channels}, eps={self.eps}"
