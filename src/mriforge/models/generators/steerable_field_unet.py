"""Steerable C_4-equivariant cross-field synthesiser (MICCAI MRIxFields2026, B-1.4).

A fully-convolutional generator that is EXACTLY equivariant to 90-degree rotations of the
input image (the cyclic group ``C_4``), built from the genuine spatial P4 block family
(:mod:`mriforge.models.blocks.equivariant.steerable_p4`). Brain anatomy in MNI space is
approximately rotation-symmetric, so constraining the hypothesis class to rotation-
equivariant maps contracts the empirical Rademacher complexity by ``~1/sqrt(|G|)``
(Cohen-Welling 2016; the B-1.4 generalisation argument) — valuable in the small
travelling-volunteer paired regime.

Pipeline (``use_equivariance=True``): lift the scalar image to a P4 regular field
(``P4LiftingConv2d``), run ``n_blocks`` of (``P4GroupConv2d`` -> orientation-shared
GroupNorm -> SiLU) with an equivariance-preserving field-FiLM, then project the regular
field back to a scalar image (``p4_to_scalar``, the regular->trivial projection that keeps
``G(rot90 x) = rot90 G(x)``).

ONE-KNOB ABLATION (``use_equivariance=False``): the SAME architecture (depth, feature
width ``base*4``, norm, activation, field-FiLM) with plain :class:`~torch.nn.Conv2d` in
place of the P4 convs. Note this single knob co-varies TWO scientific axes: it toggles the
rotation prior AND changes the parameter count (orientation weight-sharing makes the
equivariant net ~|G|=4x smaller at matched width). So the width-matched control confounds
capacity with equivariance; a PARAMETER-matched plain control (smaller ``base_width``) is
needed to isolate the rotation prior from raw capacity. The two B-1.4 control arms cover
both comparisons (width-matched and parameter-matched).

Field conditioning: a per-BASE-channel FiLM (broadcast over the 4 orientations and space)
driven by the SOURCE field strength ``tau = log10(B0_source)`` — load-bearing on Task 1
(the source field varies; the target is fixed 7T) and equivariance-preserving (a scalar
per base channel commutes with the group action). Identity at init (zero-init FiLM head),
so anti-facade tests must PERTURB the FiLM before asserting it is load-bearing.

Magnitude-only (1->1); raises on complex input.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from mriforge.models.blocks.equivariant.steerable_p4 import (
    P4GroupConv2d,
    P4LiftingConv2d,
    p4_group_norm,
    p4_to_scalar,
)
from mriforge.models.registry import register_model

_GROUP_ORDER = 4  # C_4


@register_model(name="steerable_field_unet", training_mode="steerable_synthesis")
class SteerableFieldUNet(nn.Module):
    """C_4-equivariant cross-field synthesiser with a field-FiLM and an equivariance knob."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_width: int = 16,
        kernel_size: int = 3,
        n_blocks: int = 4,
        field_hidden: int = 32,
        use_equivariance: bool = True,
        use_field_conditioning: bool = True,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "SteerableFieldUNet is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        self.base_width = int(base_width)
        self.kernel_size = int(kernel_size)
        self.n_blocks = int(n_blocks)
        self.use_equivariance = bool(use_equivariance)
        self.use_field_conditioning = bool(use_field_conditioning)
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        width = self.base_width * _GROUP_ORDER  # matched feature width across both arms

        if self.use_equivariance:
            self.lift: nn.Module = P4LiftingConv2d(1, self.base_width, self.kernel_size)
            self.blocks = nn.ModuleList(
                P4GroupConv2d(self.base_width, self.base_width, self.kernel_size)
                for _ in range(self.n_blocks)
            )
            self.norms = nn.ModuleList(p4_group_norm(self.base_width) for _ in range(self.n_blocks))
            self.proj = P4GroupConv2d(self.base_width, 1, self.kernel_size)
        else:
            pad = self.kernel_size // 2
            self.lift = nn.Conv2d(1, width, self.kernel_size, padding=pad)
            self.blocks = nn.ModuleList(
                nn.Conv2d(width, width, self.kernel_size, padding=pad) for _ in range(self.n_blocks)
            )
            self.norms = nn.ModuleList(
                nn.GroupNorm(self.base_width, width, affine=False) for _ in range(self.n_blocks)
            )
            self.proj = nn.Conv2d(width, 1, self.kernel_size, padding=pad)
        self.act = nn.SiLU()

        if self.use_field_conditioning:
            # Per-base-channel (gamma, beta) from tau=log10(B0); zero-init -> identity at
            # init. Per-base (not per-orientation) keeps the FiLM equivariant. When
            # contrast conditioning is on, a per-sample contrast one-hot is appended to
            # tau — a GLOBAL label, so the FiLM stays equivariant (broadcast over space
            # and the 4 orientations).
            cond_in = contrast_sequence_dim(1, self.num_contrasts, self.use_contrast_conditioning)
            self.field_mlp = nn.Sequential(
                nn.Linear(cond_in, field_hidden),
                nn.SiLU(),
                nn.Linear(field_hidden, 2 * self.base_width),
            )
            nn.init.zeros_(self.field_mlp[-1].weight)
            nn.init.zeros_(self.field_mlp[-1].bias)

    def _film(
        self,
        h: torch.Tensor,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the equivariance-preserving per-base-channel FiLM (broadcast over the 4
        orientations and space)."""
        if not self.use_field_conditioning:
            return h
        tau = torch.log10(field_strength.reshape(-1, 1).float().clamp_min(1e-3))
        # Append the contrast one-hot (global per-sample label) when on; #15/#9 guards.
        cond = build_contrast_sequence(
            tau,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=tau.shape[0],
            device=tau.device,
            dtype=tau.dtype,
        )
        gamma, beta = self.field_mlp(cond).chunk(2, dim=-1)  # [B, base] each
        gamma = (1.0 + gamma)[:, :, None, None]  # [B, base, 1, 1]
        beta = beta[:, :, None, None]
        b, c, hh, ww = h.shape
        h = h.reshape(b, self.base_width, _GROUP_ORDER, hh, ww)
        h = h * gamma[:, :, None] + beta[:, :, None]  # broadcast over orientation axis
        return h.reshape(b, c, hh, ww)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        # field_strength is REQUIRED (no default) — matches every sibling cross-field model
        # (field_velocity_unet, field_guided_score_unet, ...) so the Tier-2 synthetic forward
        # probe, which only injects values for no-default forward params, fills it correctly
        # (a `= None` default makes the probe skip it -> a healthy model fails the gate).
        if torch.is_complex(x):
            raise ValueError("SteerableFieldUNet expects magnitude (real) input.")
        if self.use_field_conditioning and field_strength is None:
            raise ValueError(
                "SteerableFieldUNet(use_field_conditioning=True) requires field_strength "
                "(the source field, Tesla). Got None."
            )
        h = self.lift(x)
        for conv, norm in zip(self.blocks, self.norms, strict=True):
            h = self.act(norm(conv(h)))
            if field_strength is not None:
                h = self._film(h, field_strength, contrast_id)
        out = self.proj(h)
        if self.use_equivariance:
            out = p4_to_scalar(out, 1)  # regular -> scalar image (equivariant)
        return out

    def num_parameters(self) -> int:
        """Total trainable parameters (the equivariance constraint reduces this)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
