"""Channel adapter for metrics that wrap ImageNet-pretrained networks.

LPIPS, FID, KID, DISTS, perceptual losses — every metric whose backbone
is an ImageNet-pretrained CNN expects a 3-channel RGB input. MRI data is
usually 1-channel magnitude, but in this codebase it can also be:

* 2 channels (real, imag) — interleaved complex
* 4 channels (2 coils × {real, imag}, or 2 contrasts × complex, etc.)
* 8 channels (multi-coil complex)
* genuine 3-channel RGB (rare, but possible after a transform)

The previous implementations either silently truncated to the first 3
channels (``preds = preds[:, :3]`` in LPIPS) or assumed 1-channel and
fell through into Inception with the wrong shape (FID, KID). Both are
silent fallbacks of the kind CLAUDE.md item #9 forbids.

This module supplies an explicit, documented adapter:

.. code-block:: python

    from spectramr.core.metrics.channel_adapter import adapt_to_rgb, ChannelMode
    rgb = adapt_to_rgb(x, mode=ChannelMode.AUTO)   # (B, 3, H, W)

The ``AUTO`` mode mirrors the existing
``spectramr.models.losses.perceptual_loss.PerceptualLoss`` heuristic — it is
MRI-aware (treats even ``C > 1`` as interleaved real/imag pairs and
collapses via root-sum-of-squares) and is the recommended default.
Other modes are explicit and raise rather than silently massage the
input when the assumption isn't met.
"""

from __future__ import annotations

import logging
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class ChannelMode(str, Enum):
    """How to convert ``(B, C, H, W)`` data to ``(B, 3, H, W)`` for ImageNet
    pretrained backbones.

    Every mode is explicit. None silently truncates channels.
    """

    #: MRI-aware default. ``C == 1`` → replicate to 3. ``C == 3`` → as-is.
    #: Even ``C > 1`` (or already-complex) → coil-wise root-sum-of-squares
    #: magnitude, then replicate to 3. Odd ``C > 3`` raises ``ValueError``.
    AUTO = "auto"

    #: Mean across the channel dimension → 1-channel grayscale → replicate
    #: to 3 channels. Always succeeds.
    GRAYSCALE_MEAN = "grayscale_mean"

    #: Treat ``C`` channels as ``C/2`` interleaved (real, imag) pairs,
    #: combine via root-sum-of-squares magnitude, then replicate to 3.
    #: Requires even ``C`` or already-complex input. Raises otherwise.
    COMPLEX_RSS = "complex_rss"

    #: Replicate ``C == 1`` to 3 channels. Raises for any other ``C``.
    REPLICATE = "replicate"

    #: ``C == 3`` passthrough. Raises for any other ``C``. Use only when
    #: the data is genuinely 3-channel RGB.
    PASSTHROUGH = "passthrough"


def _complex_rss(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, C, H, W) with even C or complex dtype to (B, 1, H, W) RSS."""
    if x.is_complex():
        # (B, C_complex, H, W) → coil-wise RSS magnitude
        return torch.sqrt((x.abs() ** 2).sum(dim=1, keepdim=True))
    if x.shape[1] % 2 != 0:
        raise ValueError(f"complex_rss requires even C (or complex dtype); got C={x.shape[1]}")
    B, C, H, W = x.shape
    # (B, C, H, W) → (B, H, W, C//2, 2) → complex (B, H, W, C//2) → (B, C//2, H, W)
    pairs = x.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
    cplx = torch.view_as_complex(pairs).permute(0, 3, 1, 2)
    return torch.sqrt((cplx.abs() ** 2).sum(dim=1, keepdim=True))


def adapt_to_rgb(
    x: torch.Tensor,
    mode: ChannelMode | str = ChannelMode.AUTO,
) -> torch.Tensor:
    """Convert ``(B, C, H, W)`` (or complex ``(B, C_complex, H, W)``) to
    ``(B, 3, H, W)`` real-valued RGB.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``. Real or complex dtype
            both supported (complex routes through RSS).
        mode: A :class:`ChannelMode` (or its string value).

    Returns:
        ``(B, 3, H, W)`` real-valued tensor on the same device as ``x``.

    Raises:
        ValueError: when the requested mode is incompatible with the
            input's channel count. Never silently truncates.
    """
    if x.dim() != 4:
        raise ValueError(f"adapt_to_rgb requires 4-D (B, C, H, W); got shape {tuple(x.shape)}")
    if isinstance(mode, str):
        try:
            mode = ChannelMode(mode)
        except ValueError as exc:
            valid = ", ".join(m.value for m in ChannelMode)
            raise ValueError(f"Unknown channel mode '{mode}'. Valid: {valid}") from exc

    C = x.shape[1]

    if mode is ChannelMode.AUTO:
        if x.is_complex():
            return _complex_rss(x).repeat(1, 3, 1, 1)
        if C == 1:
            return x.repeat(1, 3, 1, 1)
        if C == 3:
            return x
        if C % 2 == 0:
            return _complex_rss(x).repeat(1, 3, 1, 1)
        raise ValueError(
            f"AUTO mode cannot handle odd C != 1 and != 3 (got C={C}). "
            "Pick GRAYSCALE_MEAN to average channels, or pre-process "
            "the tensor to a known shape."
        )

    if mode is ChannelMode.GRAYSCALE_MEAN:
        if x.is_complex():
            x = x.abs()
        return x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

    if mode is ChannelMode.COMPLEX_RSS:
        return _complex_rss(x).repeat(1, 3, 1, 1)

    if mode is ChannelMode.REPLICATE:
        if C != 1:
            raise ValueError(
                f"REPLICATE mode requires C == 1; got C={C}. Use AUTO or "
                "GRAYSCALE_MEAN for multi-channel data."
            )
        return x.repeat(1, 3, 1, 1)

    if mode is ChannelMode.PASSTHROUGH:
        if C != 3:
            raise ValueError(
                f"PASSTHROUGH mode requires C == 3; got C={C}. Use AUTO or "
                "explicit COMPLEX_RSS for multi-channel data."
            )
        return x

    # Unreachable (enum exhaustiveness)
    raise ValueError(f"Unhandled ChannelMode: {mode}")


__all__ = ["ChannelMode", "adapt_to_rgb"]
