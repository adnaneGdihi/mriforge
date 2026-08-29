"""Bloch quantitative-parameter bottleneck (MICCAI MRIxFields2026, B-1.8).

A physics-bottleneck generator for cross-field translation: an encoder predicts
FIELD-INVARIANT quantitative tissue maps (PD, T1, T2) from the source image, and a
DIFFERENTIABLE SPGR signal equation renders the image at the requested field. The field
enters ONLY through the **Bottomley T1 dispersion** ``T1(B0) = T1_ref·(B0/B0_ref)^alpha``
(Bottomley 1984; Rooney 2007) — so the same qMRI maps render any field, grounding
cross-field translation in physics rather than a black-box style code.

Reuses the differentiable :func:`~mriforge.models.blocks.spgr_signal.spgr_signal`
(autograd-traced steady-state SPGR). The bottleneck is the qMRI parameters, so the
field is genuinely load-bearing (changing it changes T1 -> the SPGR weighting -> the
image) and the mechanism is structural (not a learned FiLM). ``use_field_dispersion=False``
makes T1 field-independent — the one-knob ablation. Magnitude-only (1->1); raises on
complex.

Signal scaling: the steady-state SPGR equation returns signal in arbitrary units
relative to ``M0``. At a small flip angle the SPGR *weighting* ceiling is well below 1
(~0.23 at flip=15deg over the physical T1/T2 ranges), so the raw render can never reach
the bright pixels of a [0,1]-normalised target and ``clamp(0,1)`` would be inert with a
systematic L1 floor. A single GLOBAL learnable gain (``signal_scale``, positive via
softplus) maps the relative SPGR signal into the data range. It is field- and
sample-independent, so it preserves both the dispersion ablation and the
field-is-load-bearing property (it scales every render by the same constant).

Honest scope (the physics is deliberately reduced — read before interpreting results):
- The encoder is SOURCE-FIELD-AGNOSTIC: it never receives the source field strength, so
  "field-invariant qMRI maps" is an *assumption* the encoder must learn to satisfy, not
  a guarantee. Reconstructing true field-invariant PD/T1/T2 from a single magnitude
  image at an unknown field is ill-posed; the maps are best read as a physics-structured
  latent, not calibrated qMRI.
- SINGLE-CONTRAST acquisition: one fixed (TR, TE, flip) renders one SPGR contrast.
  Cross-field datasets with differing native contrasts are not modelled.
- SINGLE global dispersion exponent ``alpha`` for all tissues; real T1(B0) dispersion
  is tissue-dependent (GM/WM/CSF differ). This is a first-order approximation.
- ``T2`` is fed into the SPGR ``T2*`` slot (GRE decay is T2*, not T2); on magnitude data
  without a measured T2* map this is a proxy, not a calibrated T2*.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from mriforge.models.blocks.spgr_signal import spgr_signal
from mriforge.models.registry import register_model


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


@register_model(name="bloch_field_bottleneck", training_mode="bloch_field")
class BlochFieldBottleneck(nn.Module):
    """qMRI bottleneck + field-dependent SPGR render for cross-field translation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 48,
        ref_field: float = 3.0,
        dispersion_alpha: float = 0.34,
        tr_ms: float = 15.0,
        te_ms: float = 4.0,
        flip_deg: float = 15.0,
        t1_range_ms: tuple[float, float] = (50.0, 4000.0),
        t2_range_ms: tuple[float, float] = (5.0, 300.0),
        signal_scale_init: float = 4.0,
        use_field_dispersion: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 1 or out_channels != 1:
            raise ValueError(
                "BlochFieldBottleneck is magnitude-only (1->1); "
                f"got in={in_channels}, out={out_channels}."
            )
        if signal_scale_init <= 0.0:
            raise ValueError(f"signal_scale_init must be > 0; got {signal_scale_init}.")
        self.ref_field = float(ref_field)
        self.dispersion_alpha = float(dispersion_alpha)
        self.tr_ms = float(tr_ms)
        self.te_ms = float(te_ms)
        self.flip_rad = math.radians(float(flip_deg))
        self.t1_lo, self.t1_hi = float(t1_range_ms[0]), float(t1_range_ms[1])
        self.t2_lo, self.t2_hi = float(t2_range_ms[0]), float(t2_range_ms[1])
        self.use_field_dispersion = bool(use_field_dispersion)
        self.encoder = nn.Sequential(_conv_block(in_channels, width), _conv_block(width, width))
        self.head = nn.Conv2d(width, 3, 1)  # raw (PD, T1, T2)
        # Global learnable signal gain (softplus -> positive). Initialised so the raw
        # SPGR ceiling (~0.23 at flip=15deg) is lifted toward the [0,1] data range; stored
        # in raw (pre-softplus) form via the inverse softplus of ``signal_scale_init``.
        inv_softplus = math.log(math.expm1(float(signal_scale_init)))
        self.signal_scale_raw = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))

    def signal_scale(self) -> torch.Tensor:
        """Positive global signal gain (softplus of the learnable raw parameter)."""
        return nn.functional.softplus(self.signal_scale_raw)

    def predict_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder -> field-invariant (PD in [0,1], T1, T2 in ms within physical range)."""
        raw = self.head(self.encoder(x))
        pd = torch.sigmoid(raw[:, 0:1])
        t1 = self.t1_lo + (self.t1_hi - self.t1_lo) * torch.sigmoid(raw[:, 1:2])
        t2 = self.t2_lo + (self.t2_hi - self.t2_lo) * torch.sigmoid(raw[:, 2:3])
        return pd, t1, t2

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("BlochFieldBottleneck expects magnitude (real) input.")
        pd, t1_ref, t2 = self.predict_parameters(x)
        if self.use_field_dispersion:
            b = field_strength.reshape(-1, 1, 1, 1).float().clamp_min(1e-3)
            ratio = (b / self.ref_field) ** self.dispersion_alpha  # Bottomley T1(B0)
            t1 = t1_ref * ratio
        else:
            t1 = t1_ref
        s = spgr_signal(pd, t1, t2, self.tr_ms, self.te_ms, self.flip_rad)
        # Lift the relative SPGR signal into the [0,1] data range with the global gain,
        # then clamp (now a genuine range guard, not a dead no-op).
        s = self.signal_scale() * s
        return s.clamp(0.0, 1.0)
