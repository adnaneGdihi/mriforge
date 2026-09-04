"""Progressive-growing GAN building blocks (Karras et al., ICLR 2018).

Canonical home for the reusable primitives used by the ``progressive_gan``
generator and its discriminator:

- :class:`~spectramr.models.blocks.progressive.fade_in.FadeIn` — phase-boundary
  linear interpolant.
- :class:`~spectramr.models.blocks.progressive.pixel_norm.PixelNorm` — pixelwise
  feature-vector normalisation (generator).
- :class:`~spectramr.models.blocks.progressive.minibatch_stddev.MinibatchStdDev` —
  across-batch standard-deviation feature channel (discriminator).

Per CLAUDE.md these reusable blocks live here and are never inlined inside
the generator/discriminator modules.
"""

from __future__ import annotations

from spectramr.models.blocks.progressive.fade_in import FadeIn
from spectramr.models.blocks.progressive.minibatch_stddev import MinibatchStdDev
from spectramr.models.blocks.progressive.pixel_norm import PixelNorm

__all__ = [
    "FadeIn",
    "MinibatchStdDev",
    "PixelNorm",
]
