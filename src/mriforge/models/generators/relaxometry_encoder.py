"""Cross-field relaxometry encoder + Bloch resynthesis (MICCAI MRIxFields2026, 2.1).

Idea 2.1: the identifiable part of Task 1 is a *relaxometry* problem. An encoder
estimates reference-field quantitative maps ``(rho, T1_ref, T2)`` plus a dispersion
exponent ``beta`` from a MULTI-CONTRAST source stack (T1w/T2w/FLAIR at the source
field — J>=3 for identifiability, Proposition 3), transports ``T1`` across field by
the empirical power law
:func:`~mriforge.infrastructure.physics.dispersion.power_law_t1_transport`
(``T1(B0)=T1_ref (B0/B0_ref)^beta``), and re-evaluates the differentiable SPGR signal
equation :func:`~mriforge.models.blocks.spgr_signal.spgr_signal` at the target field.

The field is genuinely load-bearing (it enters T1 -> the SPGR weighting -> the image),
and the signal equation is a frozen differentiable layer with a learned residual left
to the companion :class:`OpaqueResidualRefiner`. Reuses the exact physics-in-model
idiom of :class:`BlochFieldBottleneck` (global learnable ``signal_scale`` gain to lift
the relative SPGR ceiling into the [0,1] data range) — this is a sibling of
``bloch_field``, not a reimplementation of the physics (pitfall #12).

Honest scope (deliberate reductions — read before interpreting results):
- The estimated maps are best read as a physics-structured latent, not calibrated
  qMRI (single-slice magnitude inversion at an unknown field is ill-posed even with
  three contrasts).
- ``T2`` feeds the SPGR ``T2*`` slot (GRE decay is T2*, a proxy on magnitude data).
- ``beta`` is a per-voxel scalar bounded to a physiological envelope; the
  ``dispersion_prior`` loss keeps it inside ``[0.3, 0.4]``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from mriforge.infrastructure.physics.dispersion import power_law_t1_transport
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


class OpaqueResidualRefiner(nn.Module):
    """High-frequency residual confined to the opaque band (internal submodule).

    The deterministic Bloch synthesis owns the identifiable band; this small CNN adds
    detail only where the measurement is uninformative. The opaque projector ``P_op``
    is a high-pass (residual minus its average-pooled low-pass), so the residual cannot
    move the low-frequency, physically-determined structure — the input-dependence /
    no-DC-blob guard (pitfall #20). Held INSIDE :class:`RelaxometryEncoder` so the
    pipeline builds one generator (no second-model wiring); the strategy blends it with
    ``residual_weight``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        context_channels: int = 3,
        width: int = 32,
        highpass_kernel: int = 7,
    ) -> None:
        super().__init__()
        if highpass_kernel % 2 == 0 or highpass_kernel < 3:
            raise ValueError(f"highpass_kernel must be odd and >= 3; got {highpass_kernel}.")
        self._hp = highpass_kernel
        self.net = nn.Sequential(
            _conv_block(in_channels + context_channels, width),
            nn.Conv2d(width, in_channels, 1),
        )

    def _highpass(self, r: torch.Tensor) -> torch.Tensor:
        k = self._hp
        low = nn.functional.avg_pool2d(r, k, stride=1, padding=k // 2)
        return r - low

    def forward(self, y_det: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self.net(torch.cat([y_det, context], dim=1))
        return self._highpass(residual)


@register_model(
    name="relaxometry_encoder",
    training_mode="bloch_synth",
    supports_contrast_conditioning=True,
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class RelaxometryEncoder(nn.Module):
    """Multi-contrast qMRI encoder + dispersion transport + SPGR resynthesis."""

    #: Registry contracts read by the bloch_synth Tier-1 checks / reporting.
    estimated_params: tuple[str, ...] = ("rho", "T1", "T2")
    output_map_units: tuple[str, ...] = ("a.u.", "ms", "ms")

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 48,
        ref_field: float = 3.0,
        tr_ms: float = 15.0,
        te_ms: float = 4.0,
        flip_deg: float = 15.0,
        t1_range_ms: tuple[float, float] = (50.0, 4000.0),
        t2_range_ms: tuple[float, float] = (5.0, 300.0),
        signal_scale_init: float = 4.0,
        learn_beta_per_tissue: bool = True,
        dispersion_beta: float = 0.34,
        dispersion_beta_bounds: tuple[float, float] = (0.15, 0.55),
        use_opaque_residual: bool = True,
        refiner_width: int = 32,
    ) -> None:
        super().__init__()
        if out_channels != 1:
            raise ValueError(
                f"RelaxometryEncoder synthesises a single magnitude image; got "
                f"out_channels={out_channels}."
            )
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1; got {in_channels}.")
        if signal_scale_init <= 0.0:
            raise ValueError(f"signal_scale_init must be > 0; got {signal_scale_init}.")
        self.in_channels = int(in_channels)
        self.ref_field = float(ref_field)
        self.tr_ms = float(tr_ms)
        self.te_ms = float(te_ms)
        self.flip_rad = math.radians(float(flip_deg))
        self.t1_lo, self.t1_hi = float(t1_range_ms[0]), float(t1_range_ms[1])
        self.t2_lo, self.t2_hi = float(t2_range_ms[0]), float(t2_range_ms[1])
        self.learn_beta_per_tissue = bool(learn_beta_per_tissue)
        self.fixed_beta = float(dispersion_beta)
        blo, bhi = float(dispersion_beta_bounds[0]), float(dispersion_beta_bounds[1])
        if not (blo < bhi):
            raise ValueError(
                f"dispersion_beta_bounds must be (lo, hi) with lo < hi; got {blo, bhi}."
            )
        self._beta_center = 0.5 * (blo + bhi)
        self._beta_span = 0.5 * (bhi - blo)

        self.encoder = nn.Sequential(_conv_block(in_channels, width), _conv_block(width, width))
        # heads: (rho, T1_ref, T2) always; +1 raw beta channel when learning.
        n_out = 4 if self.learn_beta_per_tissue else 3
        self.head = nn.Conv2d(width, n_out, 1)
        inv_softplus = math.log(math.expm1(float(signal_scale_init)))
        self.signal_scale_raw = nn.Parameter(torch.tensor(inv_softplus, dtype=torch.float32))
        self.use_opaque_residual = bool(use_opaque_residual)
        self.refiner = (
            OpaqueResidualRefiner(in_channels=1, context_channels=in_channels, width=refiner_width)
            if self.use_opaque_residual
            else None
        )
        # Exposure contract (mechanism-fires probe + reporting).
        self.last_param_maps: dict[str, torch.Tensor] | None = None
        self.last_dispersion_beta: torch.Tensor | None = None

    def signal_scale(self) -> torch.Tensor:
        return nn.functional.softplus(self.signal_scale_raw)

    def predict_parameters(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encoder -> (rho in [0,1], T1_ref, T2 in ms, beta per-voxel)."""
        if torch.is_complex(x):
            raise ValueError("RelaxometryEncoder expects magnitude (real) input.")
        raw = self.head(self.encoder(x))
        rho = torch.sigmoid(raw[:, 0:1])
        t1_ref = self.t1_lo + (self.t1_hi - self.t1_lo) * torch.sigmoid(raw[:, 1:2])
        t2 = self.t2_lo + (self.t2_hi - self.t2_lo) * torch.sigmoid(raw[:, 2:3])
        if self.learn_beta_per_tissue:
            beta = self._beta_center + self._beta_span * torch.tanh(raw[:, 3:4])
        else:
            beta = torch.full_like(rho, self.fixed_beta)
        self.last_param_maps = {
            "rho": rho.detach(),
            "T1": t1_ref.detach(),
            "T2": t2.detach(),
        }
        self.last_dispersion_beta = beta.detach().mean()
        return rho, t1_ref, t2, beta

    def render(
        self,
        params: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        field_strength: torch.Tensor,
    ) -> torch.Tensor:
        """Transport T1 to ``field_strength`` and evaluate the SPGR signal (magnitude)."""
        rho, t1_ref, t2, beta = params
        b = field_strength.reshape(-1, 1, 1, 1).float().clamp_min(1e-3)
        t1 = power_law_t1_transport(t1_ref, self.ref_field, b, beta)
        s = spgr_signal(rho, t1, t2, self.tr_ms, self.te_ms, self.flip_rad)
        return (self.signal_scale() * s).clamp(0.0, 1.0)

    def opaque_residual(self, y_det: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """High-pass residual on the opaque band (zeros when disabled)."""
        if self.refiner is None:
            return torch.zeros_like(y_det)
        return self.refiner(y_det, context)

    def forward(self, x: torch.Tensor, *, field_strength: torch.Tensor, **_: Any) -> torch.Tensor:
        """Estimate params then synthesise at the (target) ``field_strength``.

        The reconstruction pipeline injects ``field_strength = field_strength_target``,
        so a bare ``generator(x, field_strength=...)`` call (the Tier-2 probe) yields
        the full target synthesis (deterministic Bloch render + opaque-band residual).
        The bloch_synth strategy calls :meth:`predict_parameters` / :meth:`render` /
        :meth:`opaque_residual` directly so it can weight the residual and add the
        source-consistency pass.
        """
        params = self.predict_parameters(x)
        y_det = self.render(params, field_strength)
        return (y_det + self.opaque_residual(y_det, x)).clamp(0.0, 1.0)
