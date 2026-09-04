"""Field-conditioned spectral transfer operator / FNO (MICCAI MRIxFields2026, B-1.6/3.7).

An image-domain Fourier Neural Operator whose **per-mode spectral gain is modulated by
the continuous field** (the "field-token spectral multiplier", T1 primitive #19): a small
MLP maps ``tau = log10(B0)`` to a per-Fourier-mode scaling applied to the truncated modes
inside each spectral conv, so the operator's frequency response depends on the requested
field — the mechanism for cross-field (e.g. 0.1T->7T) translation in the spectral domain.

Reuses the spectral-conv structure of :class:`~spectramr.models.blocks.fno_block.SpectralConv2d`
(``rfft2`` -> complex-weight einsum on truncated modes -> ``irfft2``); these FFTs are on
REAL feature maps and are intentional/exempt from the fft2c-SSOT rule (physics.md). The
field MLP is zero-initialised so the multiplier is the identity (scale=1) at init.
``field_token_enable=False`` makes the operator field-agnostic — the one-knob ablation.
Magnitude-only (1->1); raises on complex.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.models.registry import register_model


class FieldSpectralConv2d(nn.Module):
    """Spectral conv with a field-token per-mode multiplier on the output modes."""

    def __init__(
        self,
        channels: int,
        modes1: int,
        modes2: int,
        field_hidden: int = 32,
        field_token_enable: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.field_token_enable = bool(field_token_enable)
        scale = (2.0 / (channels + channels)) ** 0.5
        self.weights_lo = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(channels, channels, modes1, modes2, 2))
        )
        self.weights_hi = nn.Parameter(
            scale * torch.view_as_complex(torch.randn(channels, channels, modes1, modes2, 2))
        )
        # Field-token MLP -> per-mode real scaling, SEPARATE gains for the low
        # (positive-freq) and high (negative-freq) H-corners — the two corners carry
        # independent spectral content (distinct weights_lo/weights_hi), so they need
        # independent field gains (a single shared gain mis-modulates half the spectrum).
        # Output 2*modes1*modes2 -> (lo, hi). Centred at 1 via zero-init (identity).
        self.field_mlp = nn.Sequential(
            nn.Linear(1, field_hidden),
            nn.SiLU(),
            nn.Linear(field_hidden, 2 * modes1 * modes2),
        )
        nn.init.zeros_(self.field_mlp[-1].weight)
        nn.init.zeros_(self.field_mlp[-1].bias)

    @staticmethod
    def _mul(inp: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inp, w)

    def forward(self, x: torch.Tensor, field_strength: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, x_ft.size(-1))
        out_ft = torch.zeros(
            b, self.channels, h, x_ft.size(-1), dtype=torch.cfloat, device=x.device
        )
        lo = self._mul(x_ft[:, :, :m1, :m2], self.weights_lo[:, :, :m1, :m2])
        hi = self._mul(x_ft[:, :, -m1:, :m2], self.weights_hi[:, :, :m1, :m2])
        if self.field_token_enable:
            tau = torch.log10(field_strength.reshape(-1, 1).float().clamp_min(1e-3))
            # per-sample per-mode gains (centred at 1), INDEPENDENT per H-corner;
            # broadcast over channels. g[:,0]=low-corner gain, g[:,1]=high-corner gain.
            g = (1.0 + self.field_mlp(tau)).view(b, 2, self.modes1, self.modes2)
            lo = lo * g[:, 0:1, :m1, :m2]
            hi = hi * g[:, 1:2, :m1, :m2]
        out_ft[:, :, :m1, :m2] = lo
        out_ft[:, :, -m1:, :m2] = hi
        return torch.fft.irfft2(out_ft, s=(h, w))


@register_model(name="field_conditioned_fno", training_mode="field_fno")
class FieldConditionedFNO(nn.Module):
    """Image-domain field-conditioned FNO for cross-field translation (magnitude-only)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 3,
        field_hidden: int = 32,
        field_token_enable: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "FieldConditionedFNO is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spectral = nn.ModuleList(
            FieldSpectralConv2d(width, modes, modes, field_hidden, field_token_enable)
            for _ in range(n_layers)
        )
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(n_layers))
        self.act = nn.GELU()
        self.project = nn.Sequential(
            nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1)
        )

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("FieldConditionedFNO expects magnitude (real) input.")
        h = self.lift(x)
        for spec, loc in zip(self.spectral, self.local, strict=False):
            h = self.act(spec(h, field_strength) + loc(h))
        return self.project(h)
