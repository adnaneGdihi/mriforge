"""PMPS Phase-3 — paired-synthesis generator (frozen orchestrator).

Composes the frozen tissue prior (Phase 1), the Bloch forward
operator, and the frozen ULF + HF corruption operators (Phase 2) into a
single forward pass that emits a dict of
``{ulf, hf, theta, p_ulf, p_hf, defensibility_flag}``.

Inputs: a sampled tissue map ``theta`` (B, 3, H, W). Acquisition
vectors ``P_ULF`` and ``P_HF`` are passed via kwargs.

Outputs: dict with the synthesised pair plus metadata. The
``defensibility_flag`` is a Boolean tensor (B,) derived from the
per-pair residual.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


@register_model(
    name="paired_synthesis_generator",
    training_mode="paired_synthesis",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=False,
    supports_contrast_conditioning=True,
)
class PairedSynthesisGenerator(nn.Module, IGenerator):
    """Frozen Bloch + corruption-operator composition for synthesis.

    The Bloch step is intentionally simple — a SPGR-shaped magnitude
    map that depends on (T1, T2, M0, TE, TR, FA, B0). For full physics
    correctness at inference time the strategy may replace this with
    :class:`DifferentiableBlochLayer` (the encoder path) — that is a
    runtime-component swap, not a schema change.

    The defensibility flag is computed from the residual between
    ``hf_synth`` and a back-Bloch-recovered ``hf_synth`` (the latter
    only differs by stochastic corruption components, so a small
    residual indicates consistency).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        ulf_corruption: nn.Module | None = None,
        hf_corruption: nn.Module | None = None,
        defensibility_threshold: float = 1.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self.ulf_corruption = ulf_corruption
        self.hf_corruption = hf_corruption
        self.defensibility_threshold = defensibility_threshold

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def name(self) -> str:
        return "paired_synthesis_generator"

    def get_parameter_count(self) -> int:
        # All sub-modules are frozen; report 0 trainable params.
        return 0

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return (input_shape[0], self._out_channels, *input_shape[2:])

    def _bloch_spgr(
        self,
        theta: torch.Tensor,
        te: float,
        tr: float,
        flip_angle_rad: float,
    ) -> torch.Tensor:
        """Minimal SPGR Bloch magnitude on (T1, T2, M0) maps."""
        t1 = theta[:, 0:1].clamp_min(1e-6)
        t2 = theta[:, 1:2].clamp_min(1e-6)
        m0 = theta[:, 2:3]
        # Normalised maps → scale to physical units for the formula.
        # Convention: t1 in normalised [0,1] → physical 10-5000 ms.
        t1_ms = 10.0 + 4990.0 * t1
        t2_ms = 1.0 + 2499.0 * t2
        sin_a = torch.sin(torch.tensor(flip_angle_rad, device=theta.device, dtype=theta.dtype))
        cos_a = torch.cos(torch.tensor(flip_angle_rad, device=theta.device, dtype=theta.dtype))
        e1 = torch.exp(-tr / t1_ms)
        e2 = torch.exp(-te / t2_ms)
        denom = (1.0 - cos_a * e1).clamp_min(1e-6)
        return m0 * sin_a * (1.0 - e1) / denom * e2

    def forward(
        self,
        x: torch.Tensor,
        *,
        p_ulf: dict[str, float] | None = None,
        p_hf: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] < 3:
            raise ValueError(
                f"PairedSynthesisGenerator expects (B, ≥3, H, W) tissue maps; got {tuple(x.shape)}."
            )
        # Default protocols if not supplied (e.g. for smoke tests).
        p_ulf = p_ulf or {"te": 80.0, "tr": 3000.0, "fa_rad": 1.5708}
        p_hf = p_hf or {"te": 80.0, "tr": 3000.0, "fa_rad": 1.5708}
        ulf_clean = self._bloch_spgr(x, p_ulf["te"], p_ulf["tr"], p_ulf["fa_rad"])
        hf_clean = self._bloch_spgr(x, p_hf["te"], p_hf["tr"], p_hf["fa_rad"])
        ulf = self.ulf_corruption(ulf_clean) if self.ulf_corruption is not None else ulf_clean
        hf = self.hf_corruption(hf_clean) if self.hf_corruption is not None else hf_clean
        # Defensibility flag: L1 residual between (frozen-Bloch hf_clean)
        # and (hf back through inverse-of-known corruption) — here, we
        # approximate with the residual between hf and hf_clean since
        # the corruption operators are non-invertible at this scope.
        residual = (hf - hf_clean).abs().mean(dim=(1, 2, 3))
        flag = residual < self.defensibility_threshold
        return {
            "ulf": ulf,
            "hf": hf,
            "theta": x,
            "residual": residual,
            "defensibility_flag": flag,
        }

    def generate(self, z: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.forward(z, **kwargs)


__all__ = ["PairedSynthesisGenerator"]
