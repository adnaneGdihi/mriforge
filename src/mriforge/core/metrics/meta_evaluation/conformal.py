"""Conformal risk control for the selected metric (Layer L5).

L1–L4 certify the *ranking* of metrics; this module bounds the *risk of the metric
actually deployed*. It is the executable shadow of the Lean module
``Sim2Rank.Conformal`` (``docs/lean/Sim2Rank/Sim2Rank/Conformal.lean``),
implementing Conformal Risk Control (Angelopoulos, Bates, Fisch, Lei, Schuster,
ICLR 2024):

* :func:`crc_admissible` is the calibration rule
  ``(∑ ℓ_i + B)/(n+1) ≤ α`` of Lean ``crc_full_sample_mean_le``.
* :func:`crc_risk_bound` is the finite-sample guarantee ``(n+1)·α/n`` of Lean
  ``crc_calibration_mean_le`` / ``conformal_risk_control``.
* :func:`select_conformal_threshold` picks ``λ̂ = inf{λ : rule holds}`` for a loss
  nonincreasing in ``λ`` (well-posed because the admissible set is an up-set —
  Lean ``crc_threshold_upset``).
* :func:`conformal_metric_selection` applies the rule per metric to issue the
  distribution-free risk badge that composes with the L4 transfer certificate.

NumPy-free, torch-free: pure ``math`` so it imports anywhere, matching
:mod:`powered`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "ConformalRiskResult",
    "conformal_metric_selection",
    "crc_admissible",
    "crc_risk_bound",
    "select_conformal_threshold",
]


def _calibration_losses(losses: Sequence[float], loss_bound: float) -> list[float]:
    """Calibration losses with crashed (non-finite) cases clamped to max loss ``B``.

    A metric that crashed on a calibration case is recorded as ``NaN``. **Dropping**
    such cases (the previous policy) is wrong: crashes are not missing-at-random —
    they concentrate on the *hardest* cases — so dropping them shrinks the
    calibration set, makes the ``(sum L + B)/(n+1) <= alpha`` rule *easier* to pass,
    and voids the distribution-free risk guarantee (a metric that fails where it
    crashes gets the tightest badge — critique A4). Instead we conservatively count a
    crashed case at the worst-case loss ``B = loss_bound``, which preserves the
    calibration-set size ``n`` and the exchangeability the CRC bound assumes.
    """
    return [x if math.isfinite(x) else loss_bound for x in losses]


def crc_admissible(losses: Sequence[float], alpha: float, *, loss_bound: float = 1.0) -> bool:
    """The CRC calibration rule ``(∑ ℓ_i + B)/(n+1) ≤ α`` (Lean ``crc_full_sample_mean_le``).

    ``B = loss_bound`` is the worst-case loss inflation for the unseen test case.
    Crashed (non-finite) calibration cases are clamped to ``B`` rather than dropped
    (see :func:`_calibration_losses`), so a metric that crashes on hard cases is
    conservatively penalised, never rewarded with an easier test. An empty
    calibration set cannot have its risk controlled and is not admissible.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    clamped = _calibration_losses(losses, loss_bound)
    n = len(clamped)
    if n == 0:
        return False
    return (sum(clamped) + loss_bound) / (n + 1) <= alpha


def crc_risk_bound(n_calib: int, alpha: float) -> float:
    """Finite-sample risk bound ``(n+1)·α/n`` (Lean ``conformal_risk_control``).

    Converges to the target ``α`` as ``n → ∞``.
    """
    if n_calib < 1:
        raise ValueError(f"n_calib must be >= 1, got {n_calib}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return (n_calib + 1) * alpha / n_calib


@dataclass
class ConformalRiskResult:
    """Per-selection conformal risk certificate."""

    threshold: float
    admissible: bool
    risk_bound: float
    calib_mean: float
    n_calib: int
    alpha: float
    loss_bound: float

    @property
    def controlled(self) -> bool:
        """True iff the selection is admissible at the target level."""
        return self.admissible


def select_conformal_threshold(
    thresholds: Sequence[float],
    loss_at_threshold: Sequence[Sequence[float]],
    alpha: float,
    *,
    loss_bound: float = 1.0,
) -> ConformalRiskResult:
    """Pick ``λ̂ = inf{λ : (∑ ℓ_i(λ) + B)/(n+1) ≤ α}`` for a nonincreasing loss.

    ``thresholds`` is an ascending grid; ``loss_at_threshold[k][i]`` is calibration
    case ``i``'s loss at ``thresholds[k]`` (nonincreasing in ``k``). The admissible
    set is an up-set (Lean ``crc_threshold_upset``), so the first admissible
    threshold is the least conservative valid choice. If none qualifies, returns the
    most conservative (last) threshold with ``admissible = False``.
    """
    if len(thresholds) != len(loss_at_threshold):
        raise ValueError("thresholds and loss_at_threshold must be aligned")
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    for k, lam in enumerate(thresholds):
        losses = loss_at_threshold[k]
        if crc_admissible(losses, alpha, loss_bound=loss_bound):
            clamped = _calibration_losses(losses, loss_bound)
            n = len(clamped)
            return ConformalRiskResult(
                threshold=lam,
                admissible=True,
                risk_bound=crc_risk_bound(n, alpha) if n else float("nan"),
                calib_mean=sum(clamped) / n if n else float("nan"),
                n_calib=n,
                alpha=alpha,
                loss_bound=loss_bound,
            )
    clamped = _calibration_losses(loss_at_threshold[-1], loss_bound)
    n = len(clamped)
    return ConformalRiskResult(
        threshold=thresholds[-1],
        admissible=False,
        risk_bound=crc_risk_bound(n, alpha) if n else float("nan"),
        calib_mean=sum(clamped) / n if n else float("nan"),
        n_calib=n,
        alpha=alpha,
        loss_bound=loss_bound,
    )


def conformal_metric_selection(
    per_metric_losses: dict[str, Sequence[float]],
    alpha: float,
    *,
    loss_bound: float = 1.0,
) -> dict[str, ConformalRiskResult]:
    """Issue the distribution-free risk badge per metric at its operating point.

    ``per_metric_losses[m]`` is metric ``m``'s calibration diagnostic losses
    ``∈ [0, B]``. A metric is *admissible* (controlled at level ``α``) iff its
    calibration rule holds; all admissible metrics carry the same finite-sample
    risk bound ``(n+1)·α/n``. Composes with the L4 transfer certificate to form
    the validation badge.
    """
    out: dict[str, ConformalRiskResult] = {}
    for m, losses in per_metric_losses.items():
        clamped = _calibration_losses(losses, loss_bound)
        n = len(clamped)
        admissible = crc_admissible(losses, alpha, loss_bound=loss_bound)
        out[m] = ConformalRiskResult(
            threshold=float("nan"),
            admissible=admissible,
            risk_bound=crc_risk_bound(n, alpha) if n else float("nan"),
            calib_mean=sum(clamped) / n if n else float("nan"),
            n_calib=n,
            alpha=alpha,
            loss_bound=loss_bound,
        )
    return out
