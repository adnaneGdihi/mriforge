"""Fade-in interpolant for progressive-growing GANs.

When a new resolution phase is introduced, the freshly-added (randomly
initialised) layers are blended into the previously-trained network via a
linear interpolant whose weight :math:`\\alpha` is annealed from 0 to 1:

.. math::

    \\mathrm{out} = (1 - \\alpha)\\,\\mathrm{old} + \\alpha\\,\\mathrm{new},

following Karras *et al.* [Karras et al., ICLR 2018]. At ``alpha == 0`` the
new layers contribute nothing (continuity with the previous phase); at
``alpha == 1`` the network behaves as the fully-grown phase.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FadeIn(nn.Module):
    """Linear blend between an upsampled previous-phase output and the new one.

    The two inputs MUST already share the same shape (the caller upsamples
    the previous-phase tensor before passing it in).
    """

    def forward(self, old: torch.Tensor, new: torch.Tensor, alpha: float) -> torch.Tensor:
        """Blend ``old`` and ``new`` by ``alpha``.

        Args:
            old: Upsampled output of the previous (lower-resolution) phase,
                shape ``[B, C, H, W]``.
            new: Output of the newly-added phase, same shape as ``old``.
            alpha: Fade-in coefficient in ``[0, 1]``.

        Returns:
            ``(1 - alpha) * old + alpha * new``.

        Raises:
            ValueError: If ``alpha`` is outside ``[0, 1]`` or shapes differ.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"FadeIn alpha must be in [0, 1], got {alpha}")
        if old.shape != new.shape:
            raise ValueError(
                f"FadeIn requires matching shapes, got old={tuple(old.shape)} "
                f"new={tuple(new.shape)}"
            )
        return torch.lerp(old, new, alpha)
