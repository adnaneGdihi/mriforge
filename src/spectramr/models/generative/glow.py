"""Glow: multi-scale normalizing flow with invertible 1x1 convolutions.

Implements the generative flow of Kingma & Dhariwal as a composition of
``num_scales`` scales, each made of ``num_steps`` flow steps. A flow step
is (i) ActNorm, (ii) invertible 1x1 conv (LU parameterised), (iii) affine
coupling. Between scales a ``Squeeze2x`` trades spatial resolution for
channels and a ``Split`` emits half the channels to the standard-normal
prior — the multi-scale latent.

The change-of-variables likelihood is
``log p_X(x) = log p_Z(f(x)) + log|det J_f(x)|``.

Reference: D. P. Kingma and P. Dhariwal, "Glow: Generative flow with
invertible 1x1 convolutions," NeurIPS 2018.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectramr.models.blocks.flow.actnorm import ActNorm2d
from spectramr.models.blocks.flow.affine_coupling import AffineCoupling
from spectramr.models.blocks.flow.invertible_1x1_conv import InvertibleConv1x1LU
from spectramr.models.blocks.flow.squeeze import Split, Squeeze2x
from spectramr.models.registry import register_model


class _FlowStep(nn.Module):
    """One flow step: ActNorm -> InvConv1x1 -> AffineCoupling."""

    def __init__(self, num_channels: int, hidden_channels: int, coupling_type: str) -> None:
        super().__init__()
        self.actnorm = ActNorm2d(num_channels)
        self.invconv = InvertibleConv1x1LU(num_channels)
        self.coupling = AffineCoupling(num_channels, hidden_channels, coupling_type)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            x, ld_c = self.coupling(x, reverse=True)
            x, ld_w = self.invconv(x, reverse=True)
            x, ld_a = self.actnorm(x, reverse=True)
            return x, ld_a + ld_w + ld_c
        x, ld_a = self.actnorm(x, reverse=False)
        x, ld_w = self.invconv(x, reverse=False)
        x, ld_c = self.coupling(x, reverse=False)
        return x, ld_a + ld_w + ld_c


class _GlowScale(nn.Module):
    """A stack of ``num_steps`` flow steps at one resolution."""

    def __init__(
        self,
        num_channels: int,
        num_steps: int,
        hidden_channels: int,
        coupling_type: str,
    ) -> None:
        super().__init__()
        self.steps = nn.ModuleList(
            [_FlowStep(num_channels, hidden_channels, coupling_type) for _ in range(num_steps)]
        )

    def forward(self, x: torch.Tensor, reverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        log_det = x.new_zeros(x.shape[0])
        steps = reversed(self.steps) if reverse else self.steps
        for step in steps:
            x, ld = step(x, reverse=reverse)
            log_det = log_det + ld
        return x, log_det


@register_model(
    name="glow",
    training_mode="generative",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=False,
)
class Glow(nn.Module):
    """Multi-scale Glow normalizing flow.

    Args:
        in_channels: Channels of the input image (default 1, magnitude MRI).
        out_channels: Reconstruction channels; for a flow this matches
            ``in_channels`` and is accepted only for factory compatibility.
        num_scales: Number of multi-scale levels ``L``.
        num_steps: Flow steps ``K`` per scale.
        hidden_channels: Width of the coupling sub-networks.
        coupling_type: ``"affine"`` or ``"additive"``.

    Raises:
        ValueError: If ``out_channels != in_channels`` (flows are
            bijections — no silent channel change).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_scales: int = 4,
        num_steps: int = 32,
        hidden_channels: int = 512,
        coupling_type: str = "affine",
    ) -> None:
        super().__init__()
        if out_channels != in_channels:
            raise ValueError(
                "Glow is a bijection; out_channels must equal in_channels "
                f"(got in={in_channels}, out={out_channels})."
            )
        if num_scales < 1:
            raise ValueError(f"num_scales must be >= 1, got {num_scales}.")
        if coupling_type not in ("affine", "additive"):
            raise ValueError(
                f"Unknown coupling_type {coupling_type!r}; expected 'affine' or 'additive'."
            )

        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.num_steps = int(num_steps)

        self.squeeze = Squeeze2x()
        self.split = Split()

        # After each squeeze the channel count is x4; after each split
        # (all but the last scale) it is halved.
        scales: list[_GlowScale] = []
        c = self.in_channels
        for s in range(self.num_scales):
            c = c * 4  # squeeze
            scales.append(_GlowScale(c, self.num_steps, hidden_channels, coupling_type))
            if s < self.num_scales - 1:
                c = c // 2  # split keeps half
        self.scales = nn.ModuleList(scales)
        # Channel count of each emitted latent (split halves + final).
        self._latent_channels = c  # channels of the final scale output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward direction ``x -> (z, log_det_J)``.

        Returns the flattened concatenated latent and the per-sample
        accumulated log-Jacobian-determinant ``[B]``.
        """
        log_det = x.new_zeros(x.shape[0])
        zs: list[torch.Tensor] = []
        for s, scale in enumerate(self.scales):
            x = self.squeeze(x, reverse=False)
            x, ld = scale(x, reverse=False)
            log_det = log_det + ld
            if s < self.num_scales - 1:
                x, z_split = self.split(x)
                zs.append(z_split)
        zs.append(x)
        z = torch.cat([t.flatten(1) for t in zs], dim=1)
        return z, log_det

    def _scale_shapes(self, x_shape: torch.Size) -> list[torch.Size]:
        """Compute the [B,C,H,W] shape of the tensor entering each scale's split."""
        b, c, h, w = x_shape
        shapes: list[torch.Size] = []
        for s in range(self.num_scales):
            h, w = h // 2, w // 2
            c = c * 4
            if s < self.num_scales - 1:
                # split: emitted half has c//2 channels at (h, w)
                emit_c = c // 2
                shapes.append(torch.Size((b, emit_c, h, w)))
                c = c // 2
            else:
                shapes.append(torch.Size((b, c, h, w)))
        return shapes

    def inverse(self, z: torch.Tensor, input_shape: torch.Size) -> torch.Tensor:
        """Sampling direction ``z -> x``.

        Args:
            z: Flattened latent ``[B, D]`` as returned by :meth:`forward`.
            input_shape: The ``[B, C, H, W]`` shape of the original image,
                needed to reshape the flat latent back into the scale
                hierarchy.
        """
        shapes = self._scale_shapes(input_shape)
        # Recover the per-scale latent tensors from the flat vector.
        chunks: list[torch.Tensor] = []
        offset = 0
        b = z.shape[0]
        for shp in shapes:
            n = int(shp[1] * shp[2] * shp[3])
            chunks.append(z[:, offset : offset + n].view(b, shp[1], shp[2], shp[3]))
            offset += n

        # Reverse pass: start from the final scale latent.
        x = chunks[-1]
        for s in reversed(range(self.num_scales)):
            if s < self.num_scales - 1:
                x = self.split.inverse(x, chunks[s])
            x, _ = self.scales[s](x, reverse=True)
            x = self.squeeze(x, reverse=True)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample log-likelihood ``[B]`` under the standard-normal prior."""
        z, log_det_j = self.forward(x)
        d = z.shape[1]
        prior = -0.5 * (z**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)
        return prior + log_det_j
