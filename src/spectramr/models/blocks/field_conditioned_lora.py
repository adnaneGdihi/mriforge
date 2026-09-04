"""Field-conditioned low-rank (LoRA) conv modulation (MICCAI MRIxFields2026, B-3.6).

A conv whose effective weight is the shared backbone ``W_0`` plus a CONTINUOUS low-rank
update generated as a smooth function of the field pair ``(b_s, b_t)``::

    W(b_s, b_t) = W_0 + B @ diag(alpha(b_s, b_t)) @ A,   rank r << dim(W)

This is the B-3.6 "compliance-by-design" mechanism: because ``r << dim(W)`` the reachable
weight set is an ``r``-dimensional smooth manifold around ``W_0``, so the model provably CANNOT
represent independent field-specific networks (it is one shared backbone with a low-dimensional
continuous perturbation, not a disguised ensemble — the Task-3 single-model condition).

Implemented as two convs to avoid per-sample weight materialisation (which would break batched
``F.conv2d``): ``out = conv(x, W_0) + lora_up(alpha(field) ⊙ lora_down(x))`` where ``alpha`` is a
per-sample ``[B, r]`` vector from a field-MLP that scales the ``r`` intermediate channels.
``lora_up`` is zero-initialised so the update is 0 at init (the backbone alone). ``rank=0``
drops the adapter entirely -> a field-BLIND shared backbone (the one-knob ablation control).

Distinct from the FiLM cohort (B-3.5/3.8/1.4): FiLM modulates ACTIVATIONS (per-channel
scale/shift); this modulates WEIGHTS via a low-rank field-conditioned delta — a richer,
weight-space adapter. The standard (non-conditional) LoRA primitive lives in
:mod:`spectramr.models.lora`; this is the field-conditioned conv complement.
"""

from __future__ import annotations

import torch
from torch import nn


class FieldConditionedLoRAConv2d(nn.Module):
    """Conv with a field-conditioned low-rank weight modulation (B-3.6)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        rank: int = 4,
        field_dim: int = 2,
    ) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError(f"rank must be >= 0; got {rank}.")
        self.rank = int(rank)
        pad = kernel_size // 2
        self.base = nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad)
        if self.rank > 0:
            self.lora_down = nn.Conv2d(in_channels, self.rank, kernel_size, padding=pad, bias=False)
            self.lora_up = nn.Conv2d(self.rank, out_channels, 1, bias=False)
            nn.init.zeros_(self.lora_up.weight)  # delta = 0 at init (backbone alone)
            # alpha(field) -> per-rank-component coefficients, smooth in (b_s, b_t).
            # bias=False is LOAD-BEARING: with a bias, W_alpha->0 would leave a
            # NON-ZERO but field-INVARIANT delta (alpha == b_alpha), so a naive
            # "delta != 0" mechanism-fires probe would FALSE-PASS while the field
            # is silently ignored (the pitfall #16 facade this design claims
            # immunity to). Without a bias, the only field-invariant alpha is
            # alpha == 0, which coincides with delta == 0 — so "non-zero delta"
            # structurally implies "field-dependent". (This does NOT by itself
            # prevent a trained W_alpha->0 collapse toward the rank-0 backbone;
            # the LoRAModulationStrategy's `val_field_sensitivity` monitor catches
            # that residual case.)
            self.alpha_mlp = nn.Linear(field_dim, self.rank, bias=False)

    def forward(self, x: torch.Tensor, field: torch.Tensor | None = None) -> torch.Tensor:
        out = self.base(x)
        if self.rank > 0:
            if field is None:
                raise ValueError("FieldConditionedLoRAConv2d(rank>0) requires `field`.")
            alpha = self.alpha_mlp(field.float())  # [B, rank]
            lora = self.lora_up(alpha[:, :, None, None] * self.lora_down(x))  # weight-space delta
            out = out + lora
        return out
