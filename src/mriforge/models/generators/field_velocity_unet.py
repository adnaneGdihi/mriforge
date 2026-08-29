"""Field-axis velocity field for neural-ODE cross-field translation (B-3.1).

``FieldVelocityUNet`` predicts the velocity :math:`v_\\theta(x, b)` of the flow
along the field coordinate :math:`\\tau = \\log_{10} B_0`. At integration step the
network sees the current image ``x`` and the current field strength ``b`` (Tesla,
``= 10^{\\tau}``) and returns ``dx/ds`` (same shape as ``x``). Translation between
two fields is then the integral of this single velocity field — one parameter set
realises every ordered field pair, and Picard-Lindelof gives exact path-composition
(``b_s → b_m → b_t = b_s → b_t``).

Magnitude-only (1→1); raises on complex input. The field enters via
:class:`FieldFiLMBlock` (identity-init), so the field path is a genuine, non-inert
conditioning channel (anti-facade, pitfall #16).

Contrast conditioning (opt-in). By default the velocity conditions on :math:`B_0`
alone and the FiLM sequence slot is a zero vector. When ``use_contrast_conditioning``
is set, the batch's ``contrast_id`` (one of T1w/T2w/T2-FLAIR — mirrors ``_CONTRAST_INDEX``
in ``data.datasets.mrixfields_dataset``) is one-hot-encoded into that sequence slot so
the model stops mode-averaging across contrasts. The flag is validated (raises when set
without a ``contrast_id``, #15) and out-of-range ids raise (no silent clamp, #9).
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

# Contrast vocabulary shared with the mrixfields dataset's ``_CONTRAST_INDEX``
# (T1w, T2w, T2-FLAIR). Kept as a default, not imported, to preserve the
# models←data layer direction.
_NUM_CONTRASTS_DEFAULT = 3


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.SiLU(),
    )


@register_model(name="field_velocity_unet", training_mode="field_flow")
class FieldVelocityUNet(nn.Module):
    """Field-conditioned velocity predictor for field-flow (magnitude-only)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = _NUM_CONTRASTS_DEFAULT,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "FieldVelocityUNet is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if use_contrast_conditioning and num_contrasts < 2:
            raise ValueError(
                f"use_contrast_conditioning=True needs num_contrasts >= 2; got {num_contrasts}."
            )
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        self.body_in = _conv_block(in_channels, width)
        # FiLM sequence slot: contrast one-hot when conditioning is on, else a
        # scalar zero placeholder (B0-only). Shared with every field-FiLM
        # generator via models.blocks.contrast_conditioning.
        seq_dim = contrast_sequence_dim(0, self.num_contrasts, self.use_contrast_conditioning)
        self.film = FieldFiLMBlock(num_channels=width, sequence_dim=seq_dim)
        self.body_out = nn.Sequential(
            _conv_block(width, width),
            nn.Conv2d(width, out_channels, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("FieldVelocityUNet expects magnitude (real) input.")
        h = self.body_in(x)
        b = field_strength.reshape(-1, 1).float()
        # Contrast one-hot into the FiLM sequence slot when conditioning is on;
        # raises on missing (#15) / out-of-range (#9) ids, no host sync.
        seq = build_contrast_sequence(
            None,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=b.shape[0],
            device=b.device,
            dtype=b.dtype,
        )
        h = self.film.apply_to(h, b, seq)
        return self.body_out(h)
