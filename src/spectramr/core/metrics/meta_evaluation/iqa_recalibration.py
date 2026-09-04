r"""In-domain IQA recalibration (SOTA plan T5, exploratory).

T5 replaces the REJECTed zero-shot foundation IQA (CLIP-IQA / Q-Align — no
radiologist-anchored MRI validation, natural-image domain shift) with a
**monotone recalibration** of an existing score (a foundation IQA value or one of
the ~41 no-reference battery metrics) onto MRIQC- or radiologist-anchored labels:

* ``platt`` — affine ``a·s + b`` (least squares);
* ``isotonic`` — monotone non-decreasing fit (pool-adjacent-violators), which
  fixes any monotone miscalibration without assuming linearity.

Deployment is **accepted only if** the held-out Kendall-τ between the recalibrated
score and the labels clears a preregistered threshold (``accept_recalibration``).
**Corner case:** the identity map recovers the *failing* zero-shot metric, so a
non-trivial fit clearing the held-out τ gate is exactly what justifies using the
metric at all.

This is a calibration utility (parallel to ``krcps``); it is **exploratory** —
the acceptance decision needs an MRIQC/radiologist-anchored label set, which is
not shipped. Reuses ``scipy.stats.kendalltau``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import kendalltau as _scipy_kendalltau  # type: ignore[import-untyped]

_F = NDArray[np.float64]

__all__ = [
    "RecalibrationFit",
    "RecalibrationResult",
    "accept_recalibration",
    "fit_recalibration",
    "kendall_tau",
    "recalibrate",
]


@dataclass
class RecalibrationFit:
    """A fitted monotone recalibration map (``platt`` or ``isotonic``)."""

    method: str
    x_knots: _F | None = None  # isotonic: sorted scores
    y_knots: _F | None = None  # isotonic: PAVA-fitted labels
    a: float | None = None  # platt slope
    b: float | None = None  # platt intercept


@dataclass
class RecalibrationResult:
    """Held-out acceptance verdict for a recalibration."""

    accepted: bool
    heldout_tau: float
    tau_threshold: float


def kendall_tau(x: _F, y: _F) -> float:
    """Kendall's τ rank correlation (nan → 0 for degenerate/constant inputs)."""
    tau, _ = _scipy_kendalltau(np.asarray(x, float), np.asarray(y, float))
    return 0.0 if np.isnan(tau) else float(tau)  # nan (constant input) → 0


def _pava(y: _F) -> _F:
    """Pool-adjacent-violators: least-squares monotone non-decreasing fit."""
    vals: list[float] = []
    cnts: list[int] = []
    for v in np.asarray(y, float):
        cur_v, cur_c = float(v), 1
        while vals and vals[-1] > cur_v:
            pv, pc = vals.pop(), cnts.pop()
            cur_v = (cur_v * cur_c + pv * pc) / (cur_c + pc)
            cur_c += pc
        vals.append(cur_v)
        cnts.append(cur_c)
    out = np.empty(len(y), dtype=float)
    idx = 0
    for v, c in zip(vals, cnts, strict=True):
        out[idx : idx + c] = v
        idx += c
    return out


def fit_recalibration(scores: _F, labels: _F, method: str = "isotonic") -> RecalibrationFit:
    """Fit a monotone recalibration map from ``scores`` to ``labels``.

    Args:
        scores: raw metric values (calibration split).
        labels: anchored quality labels (same length).
        method: ``"isotonic"`` (PAVA monotone fit) or ``"platt"`` (affine).
    """
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must have equal shape")
    if method == "platt":
        design = np.vstack([scores, np.ones_like(scores)]).T
        a, b = np.linalg.lstsq(design, labels, rcond=None)[0]
        return RecalibrationFit(method="platt", a=float(a), b=float(b))
    if method == "isotonic":
        order = np.argsort(scores, kind="stable")
        return RecalibrationFit(
            method="isotonic",
            x_knots=scores[order],
            y_knots=_pava(labels[order]),
        )
    raise ValueError(f"method must be 'isotonic' or 'platt', got {method!r}")


def recalibrate(fit: RecalibrationFit, scores: _F) -> _F:
    """Apply a fitted recalibration map to new ``scores``."""
    scores = np.asarray(scores, float)
    if fit.method == "platt":
        a, b = fit.a, fit.b
        assert a is not None and b is not None  # narrowed for type-checkers
        return a * scores + b
    xk, yk = fit.x_knots, fit.y_knots
    assert xk is not None and yk is not None
    return np.interp(scores, xk, yk)


def accept_recalibration(
    fit: RecalibrationFit,
    heldout_scores: _F,
    heldout_labels: _F,
    tau_threshold: float,
) -> RecalibrationResult:
    """Accept the recalibration iff held-out Kendall-τ ≥ ``tau_threshold``."""
    cal = recalibrate(fit, heldout_scores)
    tau = kendall_tau(cal, heldout_labels)
    return RecalibrationResult(
        accepted=tau >= tau_threshold, heldout_tau=tau, tau_threshold=tau_threshold
    )
