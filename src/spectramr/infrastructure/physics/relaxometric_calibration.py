"""Contrast transfer between field strengths, calibrated on a known fiducial.

ULF-to-HF translation changes voxel intensities for two reasons at once:
**structure** (the low-field acquisition resolved less) and **contrast** (T1
depends on B0, so the same sequence produces a different tissue ordering at
64 mT than at 3 T). A network trained end to end learns one intensity map that
entangles both, and nothing in the output says whether the contrast half is
physically right — a plausible image can carry a wrong tissue ordering.

A fiducial with *declared* relaxometry fixes the contrast half analytically. It
is a synthetic phantom, so its ``(T1, T2)`` are not estimated but stipulated,
and the steady-state SPGR signal it produces at each field is then a closed
form. The ratio

.. math::

    \\kappa = S(\\mathrm{target}) / S(\\mathrm{source})

is a known constant, computed from the two acquisitions' ``(TR, TE, \\alpha,
B_0)`` and the marker's stipulated relaxometry. Nothing is learned.

Constraining the network to the factored form :math:`\\hat y = \\kappa\\,
g_\\theta(x)` therefore hands it the contrast transfer and leaves it only the
structural problem. The factorisation is identified up to a global scalar: with
:math:`\\kappa` fixed, an intensity error must appear in :math:`g` rather than
masquerading as contrast, and the marker gives a direct read on whether
:math:`\\kappa` was right, because on marker support the true ratio is
measurable from the data (:func:`measured_gain`).

The tissue table is the Bottomley/Marques reconciled set already used elsewhere
in the package (:mod:`spectramr.infrastructure.physics.relaxation_priors`), so
this module adds no second source of relaxometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.relaxation_priors import (
    TissueClass,
    bottomley_t1,
)

__all__ = [
    "AcquisitionParams",
    "measured_gain",
    "relaxometric_gain",
    "spgr_signal",
    "tissue_gain_map",
]


@dataclass(frozen=True)
class AcquisitionParams:
    """One SPGR acquisition, as declared by an arm.

    Attributes:
        field_strength_t: Static field in tesla. Drives ``T1`` through the
            Bottomley relation, which is the whole reason contrast transfers.
        tr_ms: Repetition time, milliseconds.
        te_ms: Echo time, milliseconds.
        flip_deg: Nominal flip angle, degrees.
    """

    field_strength_t: float
    tr_ms: float
    te_ms: float
    flip_deg: float

    def __post_init__(self) -> None:
        if self.field_strength_t <= 0.0:
            raise ValueError(f"field_strength_t must be > 0 T, got {self.field_strength_t}")
        if self.tr_ms <= 0.0 or self.te_ms <= 0.0:
            raise ValueError(f"TR and TE must be > 0 ms, got {self.tr_ms}/{self.te_ms}")
        if self.te_ms >= self.tr_ms:
            raise ValueError(
                f"TE={self.te_ms} ms must be shorter than TR={self.tr_ms} ms; the "
                "echo is read within the repetition."
            )
        if not 0.0 < self.flip_deg <= 180.0:
            raise ValueError(f"flip_deg must be in (0, 180], got {self.flip_deg}")


def spgr_signal(t1_ms: float, t2_ms: float, acq: AcquisitionParams) -> float:
    """Steady-state SPGR magnitude for unit proton density.

    .. math::

        S = \\sin\\alpha \\,\\frac{1 - E_1}{1 - \\cos\\alpha\\, E_1}\\, E_2,
        \\quad E_1 = e^{-TR/T_1},\\ E_2 = e^{-TE/T_2}

    The same expression :class:`~spectramr.infrastructure.physics.differentiable_bloch.DifferentiableBlochLayer`
    evaluates per voxel, in scalar form: here both relaxation times are declared
    rather than mapped, so there is nothing to evaluate per voxel.
    """
    if t1_ms <= 0.0 or t2_ms <= 0.0:
        raise ValueError(f"T1 and T2 must be > 0 ms, got {t1_ms}/{t2_ms}")
    alpha = math.radians(acq.flip_deg)
    e1 = math.exp(-acq.tr_ms / t1_ms)
    e2 = math.exp(-acq.te_ms / t2_ms)
    return math.sin(alpha) * (1.0 - e1) / (1.0 - math.cos(alpha) * e1) * e2


def relaxometric_gain(
    t1_source_ms: float,
    t1_target_ms: float,
    t2_ms: float,
    source: AcquisitionParams,
    target: AcquisitionParams,
) -> float:
    """``kappa = S(target) / S(source)`` for one declared tissue.

    Args:
        t1_source_ms: ``T1`` at the source field.
        t1_target_ms: ``T1`` at the target field. Distinct from the source
            value because that field dependence IS the contrast transfer; a
            single ``T1`` for both fields would make ``kappa`` a pure sequence
            ratio and the mechanism inert.
        t2_ms: ``T2``, taken as field-independent over 64 mT to 3 T, which is
            the standard approximation for brain parenchyma and is stated here
            rather than buried.
        source: The acquisition being translated FROM.
        target: The acquisition being translated TO.

    Returns:
        The multiplicative contrast transfer factor.
    """
    s_src = spgr_signal(t1_source_ms, t2_ms, source)
    if s_src <= 0.0:
        raise ValueError(
            f"source signal is {s_src}, so the gain is undefined. Check the "
            "declared TR/TE/flip against the tissue's relaxation times."
        )
    return spgr_signal(t1_target_ms, t2_ms, target) / s_src


def tissue_gain_map(
    source: AcquisitionParams,
    target: AcquisitionParams,
    t2_by_tissue: dict[TissueClass, float] | None = None,
) -> dict[TissueClass, float]:
    """``kappa`` per brain tissue class, with ``T1`` from the Bottomley relation.

    That the three gains DIFFER is what makes this a contrast change rather
    than a global scale: a single number could be absorbed by any normalisation
    and would leave the network nothing to be handed.
    """
    t2 = t2_by_tissue or {
        TissueClass.WHITE_MATTER: 70.0,
        TissueClass.GRAY_MATTER: 85.0,
        TissueClass.CSF: 1500.0,
    }
    return {
        tissue: relaxometric_gain(
            bottomley_t1(source.field_strength_t, tissue),
            bottomley_t1(target.field_strength_t, tissue),
            t2[tissue],
            source,
            target,
        )
        for tissue in TissueClass
    }


def measured_gain(
    target: torch.Tensor,
    source: torch.Tensor,
    support: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Empirical ``kappa`` on a support region: a support-weighted least squares.

    Solves ``min_k || w (k*source - target) ||^2``, which is the estimator a
    per-voxel ratio only approximates — and unlike a ratio it does not blow up
    where the source is near zero.

    Args:
        target: ``[B, C, *spatial]`` the signal being translated TO.
        source: ``[B, C, *spatial]`` the signal being translated FROM.
        support: ``[B, 1, *spatial]`` non-negative weights. On the fiducial this
            is the marker's own footprint, where the declared relaxometry
            actually holds; everywhere else the tissue is unknown and the ratio
            means nothing.

    Returns:
        ``[B]`` gains. Differentiable in both inputs.
    """
    if target.shape != source.shape:
        raise ValueError(
            f"target {tuple(target.shape)} and source {tuple(source.shape)} must match"
        )
    if support.shape[0] != target.shape[0] or support.shape[2:] != target.shape[2:]:
        raise ValueError(
            f"support {tuple(support.shape)} must share batch and spatial dims "
            f"with target {tuple(target.shape)}"
        )
    if float(support.min()) < 0.0:
        raise ValueError("support weights must be non-negative")
    reduce = tuple(range(1, target.ndim))
    w = support
    num = (w * source * target).sum(dim=reduce)
    den = (w * source * source).sum(dim=reduce)
    return num / den.clamp(min=eps)
