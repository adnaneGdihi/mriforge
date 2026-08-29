"""Octave-convolution U-Net generator.

Registered as ``octave_conv`` (image -> image reconstruction). The model
is a compact U-Net whose convolutional stages are octave convolutions
(:class:`mriforge.models.blocks.octave_conv.OctaveConv2d`), so each stage
carries a high-frequency and a low-frequency stream. The first stage
splits the dense image into the two octave streams and the final stage
merges them back into a dense image.

The reusable octave block lives in ``models/blocks/`` — this file only
composes it (CLAUDE.md canonical-homes rule).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mriforge.models.blocks.octave_conv import OctaveConv2d
from mriforge.models.registry import register_model

__all__ = ["OctaveUNet"]


class _OctaveBlock(nn.Module):
    """Two octave convolutions with GroupNorm + activation per stream.

    Both the input and output of this block are an octave
    ``(high, low)`` pair with the same ``alpha``.
    """

    def __init__(self, in_channels: int, out_channels: int, alpha: float) -> None:
        super().__init__()
        self.oc1 = OctaveConv2d(in_channels, out_channels, 3, alpha, alpha)
        self.oc2 = OctaveConv2d(out_channels, out_channels, 3, alpha, alpha)
        out_h = self.oc1.out_h
        out_l = self.oc1.out_l
        # GroupNorm groups must divide the channel count; fall back to 1.
        self.norm_h = nn.GroupNorm(self._groups(out_h), out_h) if out_h > 0 else None
        self.norm_l = nn.GroupNorm(self._groups(out_l), out_l) if out_l > 0 else None
        self.act = nn.LeakyReLU(0.2, inplace=True)

    @staticmethod
    def _groups(channels: int) -> int:
        for g in (8, 4, 2, 1):
            if channels % g == 0:
                return g
        return 1

    def _activate(
        self, pair: tuple[torch.Tensor | None, torch.Tensor | None]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        y_h, y_l = pair
        if y_h is not None and self.norm_h is not None:
            y_h = self.act(self.norm_h(y_h))
        if y_l is not None and self.norm_l is not None:
            y_l = self.act(self.norm_l(y_l))
        return y_h, y_l

    def forward(
        self, x: tuple[torch.Tensor | None, torch.Tensor | None]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        h = self._activate(self.oc1(x))
        h = self._activate(self.oc2(h))
        return h


@register_model(
    name="octave_conv",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class OctaveUNet(nn.Module):
    """U-Net with octave-convolution stages for image-domain recon.

    Args:
        in_channels: Number of input image channels.
        out_channels: Number of output image channels.
        base_channels: Channel width of the first encoder stage. Deeper
            stages double this.
        alpha: Octave ratio in ``[0, 1]`` governing the high/low split.
        depth: Number of encoder/decoder stages.
        residual: When True, add the input image to the output (residual
            reconstruction).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        alpha: float = 0.5,
        depth: int = 3,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.depth = depth
        self.residual = residual

        # Stem: dense image (alpha_in=0) -> first octave pair (alpha_out=alpha)
        self.stem = OctaveConv2d(in_channels, base_channels, 3, alpha_in=0.0, alpha_out=alpha)

        chans = [base_channels * (2**i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList(
            _OctaveBlock(chans[i], chans[i + 1], alpha) for i in range(depth)
        )
        self.decoders = nn.ModuleList(
            # decoder input = upsampled deeper feature + skip feature (concat)
            _OctaveBlock(chans[i + 1] + chans[i], chans[i], alpha)
            for i in reversed(range(depth))
        )

        # Head: octave pair (alpha_in=alpha) -> dense image (alpha_out=0)
        self.head = OctaveConv2d(base_channels, out_channels, 1, alpha_in=alpha, alpha_out=0.0)

    @staticmethod
    def _down(
        pair: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        y_h, y_l = pair
        if y_h is not None:
            y_h = F.avg_pool2d(y_h, 2)
        if y_l is not None:
            y_l = F.avg_pool2d(y_l, 2)
        return y_h, y_l

    @staticmethod
    def _up(
        pair: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        y_h, y_l = pair
        if y_h is not None:
            y_h = F.interpolate(y_h, scale_factor=2, mode="bilinear", align_corners=False)
        if y_l is not None:
            y_l = F.interpolate(y_l, scale_factor=2, mode="bilinear", align_corners=False)
        return y_h, y_l

    @staticmethod
    def _concat(
        a: tuple[torch.Tensor | None, torch.Tensor | None],
        b: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        a_h, a_l = a
        b_h, b_l = b
        y_h = a_h if b_h is None else (b_h if a_h is None else torch.cat([a_h, b_h], dim=1))
        y_l = a_l if b_l is None else (b_l if a_l is None else torch.cat([a_l, b_l], dim=1))
        return y_h, y_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image from a dense image tensor.

        Args:
            x: Input image of shape ``[B, in_channels, H, W]``.

        Returns:
            Reconstructed image of shape ``[B, out_channels, H, W]``.
        """
        feat = self.stem(x)

        skips: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []
        for enc in self.encoders:
            # Append the stage's INPUT, not its output. The decoders are built
            # as `_OctaveBlock(chans[i + 1] + chans[i], chans[i])` -- i.e. the
            # upsampled deeper feature (`chans[i + 1]`) concatenated with a skip
            # of width `chans[i]`, the width BEFORE this stage widened it.
            #
            # Appending `enc(feat)` gave the skip `chans[i + 1]` instead, so the
            # concat was `2 * chans[i + 1]` and every decoder received more
            # channels than it declared: at base=16, depth=2, alpha=0.25 the
            # first decoder saw 96 high channels against a conv built for 72.
            skips.append(feat)
            feat = enc(feat)
            feat = self._down(feat)

        for dec in self.decoders:
            feat = self._up(feat)
            skip = skips.pop()
            feat = self._concat(feat, skip)
            feat = dec(feat)

        y_h, y_l = self.head(feat)
        # head has alpha_out=0 -> only the high (dense) stream is produced
        out = y_h if y_h is not None else y_l
        assert out is not None  # head always emits one stream

        if self.residual:
            if out.shape[1] == x.shape[1]:
                out = out + x
        return out
