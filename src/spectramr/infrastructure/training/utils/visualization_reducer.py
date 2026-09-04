"""One reduction from a raw model output to a displayable image (#709, #390).

Five render sites each grew their own version of the same chain --
``metrics_mixin`` (validation previews), ``train.py`` (validation PNGs),
``debug_snapshot`` (training snapshots), ``pinn_strategy`` and
``disentangled_strategy``. They disagreed in ways nobody could see from a single
figure:

* only the diffusion strategy undid ``log1p`` before an IFFT (#682);
* only ``heteroscedastic_ulf`` reduced a distribution head to its point
  estimate, so ``evidential_unet``'s ``[mean, var, alpha, beta]`` rendered as
  ``sqrt(Σ params²)`` (#390);
* the validation-PNG path windowed twice, once in ``train.py`` and again in
  ``MetricsTracker._normalize_images``, so the same tensor came out at visibly
  different contrast depending on which writer produced it.

The chain, in order, and why the order is the order:

1. **decompress** -- ``expm1`` when the pipeline applied ``log1p``. Must precede
   the IFFT: ``log1p`` is a per-bin nonlinearity, so the transform of a
   compressed spectrum is not a scaled image.
2. **reduce channels** -- collapse a *distribution* head to the quantity a human
   should look at. Must precede the magnitude step, which would otherwise blend
   distribution parameters into one number.
3. **inverse transform** -- ``ifft2c`` when the tensor is k-space.
4. **magnitude** -- one channel out.

Windowing is deliberately NOT here. It belongs to the writer, and having two
owners is what produced the double-windowing; this module ends at a
single-channel magnitude and the writer windows once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from spectramr.infrastructure.training.utils.kspace_view import (
    decompress_for_view,
    log_scaling_enabled,
)

logger = logging.getLogger(__name__)

__all__ = ["VisualizationReducer", "is_distribution_head", "to_magnitude"]


def to_magnitude(t: torch.Tensor) -> torch.Tensor:
    """Reduce a possibly-multi-channel image-domain tensor to one magnitude channel.

    Hoisted from a closure inside ``MetricsMixin`` that ``diffusion.py`` and
    ``debug_snapshot`` had each re-implemented.

    Image-domain visualization MUST NOT assume an even channel count means
    interleaved ``(R, I)``: paired-modality data (ULF/HF, T1/T2, …) is also
    even-channel, and pairing it as complex blends the modalities into a
    ``sqrt(M_a² + M_b²)`` composite -- the "doubled and odd" regression. For
    genuine real-stacked complex tensors channel-RSS is mathematically
    equivalent to the per-pair-magnitude-then-coil-RSS chain, so RSS is correct
    for both and assumption-free for neither.
    """
    if torch.is_complex(t):
        return torch.abs(t)
    if t.dim() in (4, 5) and t.shape[1] > 1:
        return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
    return t


def is_distribution_head(config: Any, strategy: Any = None) -> bool:
    """Whether the model emits distribution PARAMETERS rather than an image.

    Deliberately the *same* declaration the width guard in
    ``BaseTrainingStrategy.train_step`` consults. That guard already has to know
    which heads legitimately emit more channels than the target; the
    visualization side asked the question separately and got a different answer,
    which is why ``evidential_unet`` rendered as a parameter blend while
    ``heteroscedastic_ulf`` rendered correctly (#390). One declaration, two
    consumers.
    """
    model = getattr(config, "model", None)
    if getattr(model, "model_type", None) == "evidential_unet":
        return True
    return bool(getattr(strategy, "predicts_distribution_params", False))


@dataclass(frozen=True)
class VisualizationReducer:
    """Built once from the config; owns the raw-output → displayable chain.

    Attributes:
        log_scaled: ``data.processing.enable_log_scaling`` -- whether ``expm1``
            is owed before an inverse transform.
        distribution_head: whether channel 0 is a point estimate rather than
            part of an image.
    """

    log_scaled: bool = False
    distribution_head: bool = False

    @classmethod
    def from_config(cls, config: Any, strategy: Any = None) -> VisualizationReducer:
        """Resolve both flags from the SSOT config, once."""
        return cls(
            log_scaled=log_scaling_enabled(config),
            distribution_head=is_distribution_head(config, strategy),
        )

    # -- step 2 -----------------------------------------------------------
    def point_estimate(self, t: torch.Tensor) -> torch.Tensor:
        """The channel a human should look at.

        Channel 0 for a distribution head -- the mean, for every parametric head
        in the tree (``evidential_unet``'s ``[mean, var, alpha, beta]``,
        heteroscedastic ``[mean, logvar]``). Identity otherwise, which is
        load-bearing: a 2-channel COMPLEX tensor must reach
        :func:`to_magnitude` intact, and taking channel 0 of one would show the
        real part alone and call it the image.

        A strategy that knows better overrides
        ``MetricsMixin._prediction_for_visualization``; this is the default it
        falls back to, not a policy imposed over it.
        """
        if not self.distribution_head:
            return t
        if t.dim() >= 2 and t.shape[1] >= 1:
            return t[:, :1]
        return t

    def uncertainty(self, t: torch.Tensor) -> torch.Tensor | None:
        """The variance channel of a distribution head, or ``None``.

        Channel 1 by the same convention that makes channel 0 the mean. Returned
        separately rather than folded into the displayed image because it is a
        different quantity in different units -- rendering it *as* the image is
        the #390 defect, and averaging it into one is worse.

        The caller decides whether to show it; nothing here forces a panel.
        """
        if not self.distribution_head:
            return None
        if t.dim() >= 2 and t.shape[1] >= 2:
            return t[:, 1:2]
        return None

    # -- the whole chain --------------------------------------------------
    def to_display(
        self,
        t: torch.Tensor,
        *,
        in_kspace: bool = False,
        channel_dim: int = 1,
    ) -> torch.Tensor:
        """Raw model output → single-channel magnitude, ready to window.

        Args:
            t: Prediction, target or input, image- or k-space-domain.
            in_kspace: Whether ``t`` is a measurement. The caller decides -- this
                module does not guess a domain, because a wrong guess produces a
                plausible-but-wrong picture rather than an obvious failure.
            channel_dim: Channel axis for real-stacked complex input.

        Returns:
            A magnitude tensor. NOT windowed: the writer owns that, and two
            owners is what produced the double-windowing.
        """
        if in_kspace:
            t = decompress_for_view(t, log_scaled=self.log_scaled, channel_dim=channel_dim)
        t = self.point_estimate(t)
        if in_kspace:
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            t = ifft2c(_as_complex(t, channel_dim))
        return to_magnitude(t)


def _as_complex(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Interpret a real-stacked tensor as complex, or pass a complex one through.

    An odd channel count cannot be interleaved ``(R, I)``, so it is returned
    unchanged rather than reshaped into something that would transform cleanly
    and mean nothing.
    """
    if torch.is_complex(t):
        return t
    if t.dim() > channel_dim and t.shape[channel_dim] % 2 == 0:
        return torch.stack(
            [torch.complex(t[:, i], t[:, i + 1]) for i in range(0, t.shape[channel_dim], 2)],
            dim=channel_dim,
        )
    return t
