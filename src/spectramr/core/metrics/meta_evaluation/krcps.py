r"""K-RCPS — entrywise conformal risk control for reconstruction (SOTA plan Phase C).

Scalar conformal risk control (:mod:`.conformal`) certifies a *single* interval /
threshold for the whole image. K-RCPS (Teneggi *et al.*, "How to Trust Your
Diffusion Model: A Convex Optimization Approach to Conformal Risk Control",
ICML 2023) controls the *entrywise* (per-pixel / per-region) risk: each entry
gets its own prediction-interval half-width.

This implementation calibrates the smallest scale :math:`\beta \ge 0` over a
fixed per-entry width profile :math:`w` (e.g. a heteroscedastic uncertainty map,
or a K-group-constant profile) such that the **marginal** miscoverage CRC bound
holds:

.. math::
    \hat{\beta} = \inf\Big\{\beta : \frac{\sum_i \ell_i(\beta) + B}{n+1} \le \alpha
    \Big\}, \qquad \ell_i(\beta) = \frac{1}{D}\sum_d \mathbf{1}\big[|r_{i,d}| >
    \beta\,w_d\big].

The calibrated entrywise half-width is :math:`\beta\,w_d`. Because
:math:`\ell_i(\beta)` is nonincreasing in :math:`\beta`, the admissible set is an
up-set and the calibration reuses :func:`select_conformal_threshold` verbatim —
so K-RCPS inherits the finite-sample risk bound :math:`(n+1)\alpha/n` (Lean
``conformal_risk_control``).

**Corner case:** a uniform width profile (:math:`w_d \equiv 1`) reduces this to
the scalar conformal threshold on :math:`|r|`, so K-RCPS provably generalises the
scalar RCPS/CRC. This is RCPS with an entrywise width profile + the CRC marginal
guarantee, **not** the full convex-program interval-*volume* minimisation of
Teneggi 2023 (a future increment); the width profile is supplied, not optimised.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from spectramr.core.metrics.meta_evaluation.conformal import select_conformal_threshold

__all__ = ["KRCPSResult", "krcps_calibrate"]


@dataclass
class KRCPSResult:
    """Entrywise conformal risk certificate."""

    beta: float
    widths: list[float]
    risk_bound: float
    controlled: bool
    n_groups: int
    alpha: float
    loss_bound: float = 1.0

    @property
    def interval_widths(self) -> list[float]:
        """Calibrated per-entry interval half-widths ``β·w_d``."""
        return [self.beta * w for w in self.widths]


def _miscoverage(row: Sequence[float], widths: Sequence[float], beta: float) -> float:
    """Fraction of entries whose abs-residual exceeds the scaled width ``β·w_d``."""
    d = len(widths)
    return sum(abs(r) > beta * w for r, w in zip(row, widths, strict=True)) / d


def krcps_calibrate(
    residuals: Sequence[Sequence[float]],
    widths: Sequence[float],
    alpha: float,
    *,
    loss_bound: float = 1.0,
    n_grid: int = 200,
) -> KRCPSResult:
    r"""Calibrate the entrywise interval scale ``β`` (K-RCPS).

    Args:
        residuals: calibration abs-errors ``|x̂ - x|`` per entry, shape
            ``[n_calib, D]``.
        widths: nominal per-entry interval half-widths, shape ``[D]`` (all > 0).
            Distinct values = K groups.
        alpha: target risk level in ``(0, 1)``.
        loss_bound: the CRC loss bound ``B`` (miscoverage is in ``[0, 1]``).
        n_grid: number of ascending ``β`` candidates.

    Returns:
        A :class:`KRCPSResult` with the calibrated ``β`` (hence entrywise interval
        widths ``β·w``), the finite-sample risk bound, and whether the marginal
        risk is controlled at ``α``.

    Raises:
        ValueError: on empty/mismatched shapes, non-positive widths, or
            ``alpha`` outside ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not residuals:
        raise ValueError("residuals must be a non-empty [n_calib, D] sequence")
    d = len(widths)
    if d == 0:
        raise ValueError("widths must be non-empty")
    if any(len(row) != d for row in residuals):
        raise ValueError(f"every residual row must have length D={d} (matching widths)")
    if any(w <= 0.0 for w in widths):
        raise ValueError("all widths must be > 0")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    ratios = [abs(r) / w for row in residuals for r, w in zip(row, widths, strict=True)]
    beta_max = max(ratios) if ratios else 1.0
    if beta_max <= 0.0:
        beta_max = 1.0  # all residuals zero — any β controls risk; grid still valid.
    betas = [beta_max * k / (n_grid - 1) for k in range(n_grid)]  # ascending
    # loss_at_threshold[k][i] = miscoverage of calib sample i at β_k (nonincreasing in k).
    loss_at_threshold = [[_miscoverage(row, widths, beta) for row in residuals] for beta in betas]
    sel = select_conformal_threshold(betas, loss_at_threshold, alpha, loss_bound=loss_bound)
    return KRCPSResult(
        beta=sel.threshold,
        widths=list(widths),
        risk_bound=sel.risk_bound,
        controlled=sel.admissible,
        n_groups=len(set(widths)),
        alpha=alpha,
        loss_bound=loss_bound,
    )
