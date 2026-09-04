r"""MR spectroscopy losses (regime: ``mri_spectroscopy``).

Fitting the FID **is** MRS quantification: the metabolite concentrations are the
resonance amplitudes. So the primary objective is self-supervised — the measured
FID supplies its own supervision, and no ground-truth concentrations are needed,
which is what makes the regime trainable on real spectroscopy data. This is the
same shape as ``tofts_residual`` for perfusion and ``mrf_dictionary_match`` for
fingerprinting.

The fit is done in the **time domain**, on the FID, which is what AMARES and
QUEST do. Two reasons it is not done on the spectrum: the noise is white in time
and coloured by any apodisation applied before the transform, and a truncated FID
produces sinc wiggles in the spectrum that a spectral-domain residual would try
to fit as signal.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.signal_models.spectroscopy import lorentzian_fid
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="mrs_fid_residual",
    domain="image",
    workflows=frozenset({Regime.SPECTROSCOPIC}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class MRSFIDResidualLoss(nn.Module):
    r"""Self-supervised fit residual: does the model reproduce the measured FID?

    Args:
        forward_model: the ``(t_s, amplitudes, frequencies_hz, linewidths_hz,
            phases_rad) -> fid`` map. Defaults to :func:`lorentzian_fid`.
            ``MRSQuantificationStrategy`` passes the model it resolved from
            ``data.spectroscopy.signal_model`` via the ``SignalModelRegistry``,
            so that key dispatches for real rather than being re-hardcoded here.

    The residual is taken on the **complex** FID, over real and imaginary parts
    together. Fitting the magnitude instead would discard the phase — and phase is
    a fitted parameter, so a magnitude residual would leave it completely
    unconstrained while still converging nicely.

    **This objective is identifiable but NOT convex, and that is the opposite of
    the perfusion trap — read the residual accordingly.** Fitting a curve
    synthesised from known parameters recovers all 12 of them exactly from a
    nearby start (residual 1e-4), but the frequency capture range is only about
    +/-30 Hz:

    ==========================  ==========  ==========
    frequency init (true 120)   residual    recovered?
    ==========================  ==========  ==========
    off by 30 Hz                 1.07e-04   yes
    off by 100 Hz                1.64e-01   no
    all zeros                    1.66e-01   no (all 3 collapse onto one line)
    ==========================  ==========  ==========

    The important part is the residual column. A wrong MRS fit announces itself —
    it is **three orders of magnitude worse**. Contrast ``tofts_residual``, where
    a ~1e-4 residual happily coexists with a *negative* ``vp``: there the data
    cannot identify the parameter and the residual LIES, so a constraint is
    needed. Here the data identifies everything and the residual TELLS THE TRUTH;
    what is needed is a starting point, which is precisely why AMARES supplies
    prior knowledge of metabolite positions. So: do not add a constraint to
    "fix" a bad frequency fit — fix the initialisation, and trust the residual.
    """

    def __init__(
        self,
        forward_model: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model or lorentzian_fid

    def forward(
        self,
        t_s: torch.Tensor,
        amplitudes: torch.Tensor,
        frequencies_hz: torch.Tensor,
        linewidths_hz: torch.Tensor,
        phases_rad: torch.Tensor,
        measured_fid: torch.Tensor,
    ) -> torch.Tensor:
        predicted = self.forward_model(t_s, amplitudes, frequencies_hz, linewidths_hz, phases_rad)
        if predicted.shape != measured_fid.shape:
            raise ValueError(
                f"the modelled FID {tuple(predicted.shape)} does not match the "
                f"measured one {tuple(measured_fid.shape)}. Left to broadcast, "
                "this would compare every voxel's model against every other "
                "voxel's data and still return an ordinary number."
            )
        # torch.view_as_real keeps the gradient exact for complex tensors and
        # weights the real and imaginary residuals equally, which .abs() does not
        # (its gradient is undefined at zero and it drops the phase information
        # the fit is supposed to be constraining).
        residual = predicted - measured_fid
        if torch.is_complex(residual):
            residual = torch.view_as_real(residual)
        return residual.abs().mean()


@register_loss(
    name="mrs_prior_knowledge",
    domain="image",
    workflows=frozenset({Regime.SPECTROSCOPIC}),
)
class MRSPriorKnowledgeLoss(nn.Module):
    r"""AMARES-style soft prior knowledge on the resonance parameters.

    Penalises the physically impossible: negative amplitudes (a concentration
    cannot be negative) and linewidths outside a plausible range (a resonance
    narrower than the shim can produce, or so broad it is not a resonance).

    Args:
        min_linewidth_hz: below this a line is narrower than any real shim allows.
        max_linewidth_hz: above this the "resonance" is baseline.

    The bounds are hinges, not clamps, so they push rather than truncate and the
    gradient survives.

    **Read a healthy value here carefully.** Under the strategy's default
    ``parameter_activation="softplus"`` both amplitudes and linewidths are forced
    positive *structurally*, so the ``clamp(-amplitude)`` term is identically zero
    with zero gradient — softplus already did that job — and the only live terms
    are the linewidth bounds. This is the same trap as the perfusion box loss and
    ``vp``: a healthy prior-knowledge loss is NOT evidence that non-negativity is
    being learned. Under ``parameter_activation="none"`` it is this loss or
    nothing.
    """

    def __init__(
        self,
        min_linewidth_hz: float = 1.0,
        max_linewidth_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if not 0 < min_linewidth_hz < max_linewidth_hz:
            raise ValueError(
                "need 0 < min_linewidth_hz < max_linewidth_hz; got "
                f"{min_linewidth_hz} and {max_linewidth_hz}."
            )
        self.min_linewidth_hz = float(min_linewidth_hz)
        self.max_linewidth_hz = float(max_linewidth_hz)

    def forward(
        self,
        amplitudes: torch.Tensor,
        linewidths_hz: torch.Tensor,
    ) -> torch.Tensor:
        negative_amplitude = torch.clamp(-amplitudes, min=0.0)
        too_narrow = torch.clamp(self.min_linewidth_hz - linewidths_hz, min=0.0)
        too_broad = torch.clamp(linewidths_hz - self.max_linewidth_hz, min=0.0)
        return negative_amplitude.mean() + too_narrow.mean() + too_broad.mean()


__all__ = ["MRSFIDResidualLoss", "MRSPriorKnowledgeLoss"]
