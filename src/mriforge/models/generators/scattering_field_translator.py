"""Field-conditioned ->7T translator for the scattering-Besov detail prior (B-1.3).

A compact magnitude-only translator for MRIxFields2026 Task 1 (any lower field -> fixed 7T): a
conv backbone whose activations are FiLM-modulated by the SOURCE field strength
(:class:`~mriforge.models.blocks.field_film_modulation.FieldFiLMBlock`, zero-init -> identity
start). The B-1.3 contribution is the scattering+Besov DETAIL LOSS
(:class:`~mriforge.models.losses.scattering_besov_loss.ScatteringBesovLoss`); this backbone is the
field-conditioned host that the loss drives toward 7T-like high-frequency detail.

Task 1 target is fixed 7T, so only ``field_strength`` (the source field, Tesla) is a required
forward kwarg (no default -> the Tier-2 probe injects it); magnitude-only (1->1).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from mriforge.models.blocks.field_film_modulation import FieldFiLMBlock
from mriforge.models.registry import register_model


@register_model(name="scattering_field_translator", training_mode="scattering_besov")
class ScatteringFieldTranslator(nn.Module):
    """Source-field-conditioned ->7T magnitude translator (B-1.3 host backbone)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        n_blocks: int = 3,
        kernel_size: int = 3,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                f"ScatteringFieldTranslator is magnitude-only (1->1); got in={in_channels}, "
                f"out={out_channels}."
            )
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1; got {n_blocks}.")
        self.width = int(width)
        self.n_blocks = int(n_blocks)
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        pad = kernel_size // 2
        self.lift = nn.Conv2d(1, width, kernel_size, padding=pad)
        self.block_convs = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size, padding=pad) for _ in range(n_blocks)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(min(8, width), width) for _ in range(n_blocks))
        # FiLM sequence = log-field (dim 1) + contrast one-hot when conditioning is on.
        seq_dim = contrast_sequence_dim(1, self.num_contrasts, self.use_contrast_conditioning)
        self.films = nn.ModuleList(
            FieldFiLMBlock(num_channels=width, sequence_dim=seq_dim) for _ in range(n_blocks)
        )
        self.act = nn.SiLU()
        self.head = nn.Conv2d(width, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("ScatteringFieldTranslator expects magnitude (real) input.")
        b_s = field_strength.reshape(-1).float()
        if b_s.shape[0] != x.shape[0]:
            raise ValueError(
                f"field_strength batch {b_s.shape[0]} != input batch {x.shape[0]}; "
                "pass one source field per sample."
            )
        log_field = torch.log10(b_s.clamp_min(1e-3)).reshape(-1, 1)  # [B,1] log-field
        # Append the contrast one-hot to the log-field conditioning vector when on
        # (raises on missing/out-of-range ids, #15/#9); passthrough when off.
        seq = build_contrast_sequence(
            log_field,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=x.shape[0],
            device=x.device,
            dtype=log_field.dtype,
        )
        h = self.act(self.lift(x))
        for conv, norm, film in zip(self.block_convs, self.norms, self.films, strict=True):
            h = self.act(norm(conv(h)))
            h = film.apply_to(h, b_s, seq)  # FiLM-modulate by the source field
        return self.head(h)
