"""Field-conditioned SIREN INR for cross-field SR / translation (MICCAI, B-2.8).

A coordinate-MLP (SIREN) renders the high-field image as a continuous function of
spatial coordinates, conditioned on (a) per-coordinate **anatomy features** encoded
from the ULF source image and (b) the **continuous target field** ``b = 10**(log10 B0)``
via a learned style that FiLM-modulates the SIREN. Because the network is a coordinate
function, it is resolution-free: querying a denser grid gives super-resolution.

Composes existing primitives: :class:`SIRENWithFiLM` (the FiLM-conditioned INR decoder)
+ a small conv source encoder + a field->style MLP. Magnitude-only (1->1); raises on
complex. The field is genuine conditioning (anti-facade): the style depends on the
field and FiLM-modulates every hidden layer — so changing the target field changes the
rendered intensity (once the FiLM moves off its identity init).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.contrast_conditioning import (
    build_contrast_sequence,
    contrast_sequence_dim,
)
from spectramr.models.inr.siren import SIRENWithFiLM
from spectramr.models.registry import register_model


@register_model(
    name="field_conditioned_inr",
    training_mode="field_conditioned_inr",
    supports_contrast_conditioning=True,
)
class FieldConditionedINR(nn.Module):
    """Field-conditioned SIREN INR: (coords + source anatomy, field-style) -> image."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        feat_dim: int = 16,
        hidden_features: int = 128,
        hidden_layers: int = 4,
        style_dim: int = 64,
        use_contrast_conditioning: bool = False,
        num_contrasts: int = 3,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "FieldConditionedINR is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        self.feat_dim = int(feat_dim)
        self.use_contrast_conditioning = bool(use_contrast_conditioning)
        self.num_contrasts = int(num_contrasts)
        # Source encoder: per-coordinate anatomy features from the ULF image.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, self.feat_dim, 3, padding=1),
        )
        # Continuous field (+ contrast one-hot when on) -> style vector that FiLM-
        # modulates the SIREN.
        cond_in = contrast_sequence_dim(1, self.num_contrasts, self.use_contrast_conditioning)
        self.field_mlp = nn.Sequential(
            nn.Linear(cond_in, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )
        self.siren = SIRENWithFiLM(
            in_features=2 + self.feat_dim,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=out_channels,
            style_dim=style_dim,
            use_film=True,
        )

    @staticmethod
    def _coords(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(h * w, 2)  # [H*W, 2]

    def forward(
        self,
        x: torch.Tensor,
        *,
        field_strength: torch.Tensor,
        out_hw: tuple[int, int] | None = None,
        contrast_id: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Render the target image at the native (or, if ``out_hw`` is given, an
        arbitrary) output grid.

        ``out_hw=(H_out, W_out)`` exercises the resolution-free property: the SIREN is
        queried at the requested grid and the source anatomy features are
        ``grid_sample``-interpolated to that grid — so the same trained model renders
        at a denser lattice (super-resolution), decoupling the output resolution from
        the input lattice. ``out_hw=None`` renders at the input resolution.
        """
        if torch.is_complex(x):
            raise ValueError("FieldConditionedINR expects magnitude (real) input.")
        b = x.shape[0]
        h_out, w_out = out_hw if out_hw is not None else (x.shape[-2], x.shape[-1])
        feat = self.encoder(x)  # [B, feat_dim, H_in, W_in]
        coords = self._coords(h_out, w_out, x.device, x.dtype)  # [H_out*W_out, 2]
        coords_b = coords.unsqueeze(0).expand(b, -1, -1)  # [B, H_out*W_out, 2]
        if out_hw is None:
            feat_flat = feat.permute(0, 2, 3, 1).reshape(b, h_out * w_out, self.feat_dim)
        else:
            # Sample source features at the (possibly denser) query coords. grid_sample
            # grid is [B, H_out, W_out, 2] in (x, y) order, normalised to [-1, 1].
            grid = coords.view(1, h_out, w_out, 2).expand(b, -1, -1, -1)
            sampled = torch.nn.functional.grid_sample(
                feat, grid, mode="bilinear", align_corners=True
            )  # [B, feat_dim, H_out, W_out]
            feat_flat = sampled.permute(0, 2, 3, 1).reshape(b, h_out * w_out, self.feat_dim)
        inp = torch.cat([coords_b, feat_flat], dim=-1)  # [B, H_out*W_out, 2+feat_dim]
        # Append the contrast one-hot to the field before the style MLP when on (#15/#9).
        field_vec = field_strength.reshape(-1, 1).float()
        cond = build_contrast_sequence(
            field_vec,
            contrast_id,
            self.num_contrasts,
            self.use_contrast_conditioning,
            batch_size=b,
            device=x.device,
            dtype=field_vec.dtype,
        )
        style = self.field_mlp(cond)  # [B, style_dim]
        out = self.siren(inp, style)  # [B, H_out*W_out, 1]
        return out.reshape(b, h_out, w_out, -1).permute(0, 3, 1, 2)  # [B, 1, H_out, W_out]
