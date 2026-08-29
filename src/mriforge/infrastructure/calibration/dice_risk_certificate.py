r"""Dice-Risk RCPS Certificate (B-1.7) — risk-controlling prediction sets.

The MICCAI MRIxFields2026 binding metric is SynthSeg-Dice on 14 deep grey-matter
nuclei; distribution-matching synthesis is known to hallucinate structure there.
This certificate gives a **finite-sample, distribution-free upper bound on the
expected Dice-risk** :math:`R = 1 - \mathrm{Dice}`, following the risk-controlling
prediction set (RCPS) framework of Bates, Angelopoulos, Lei, Malik & Jordan (2021),
with a Hoeffding pointwise upper-confidence bound.

Two pieces (mirroring :mod:`pathology_recall_certificate`, the dual recall bound):

- :func:`hoeffding_risk_upper_bound` — :math:`\hat R + \sqrt{\ln(1/\delta)/(2n)}`,
  the UCB on the expected risk (clamped to ``[0, 1]``).
- :func:`rcps_threshold` — given a monotone (non-increasing) risk curve over an
  abstention/uncertainty threshold :math:`\lambda`, return the smallest
  :math:`\lambda` whose risk UCB satisfies the budget :math:`\alpha` (the RCPS
  calibration); ``inf`` if no threshold controls the risk.
- :class:`DiceRiskCertificate` / :class:`DiceRiskReport` — the cohort-level
  certify-gate the calibration strategy serialises.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass


def hoeffding_risk_upper_bound(
    empirical_risk: float, n: int, delta: float, loss_bound: float = 1.0
) -> float:
    r"""UCB on the expected risk: :math:`\hat R + b\sqrt{\ln(1/\delta)/(2n)}`.

    Args:
        empirical_risk: mean risk ``hat R`` over the calibration cohort.
        n: cohort size.
        delta: confidence parameter ``δ ∈ (0, 1)``.
        loss_bound: the range ``b`` of the bounded risk (Dice-risk ∈ [0, 1] ⇒ 1).

    Returns:
        The upper confidence bound, clamped to ``[0, loss_bound]``.
    """
    if not 0.0 <= empirical_risk <= loss_bound:
        raise ValueError(f"empirical_risk must lie in [0, {loss_bound}]; got {empirical_risk}")
    if n <= 0:
        raise ValueError(f"n must be > 0; got {n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    gap = loss_bound * math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return min(loss_bound, empirical_risk + gap)


def rcps_threshold(
    risk_curve: Sequence[tuple[float, float]],
    alpha: float,
    delta: float,
    n: int,
    loss_bound: float = 1.0,
) -> float:
    r"""RCPS calibration: smallest abstention threshold whose risk UCB ≤ ``alpha``.

    The risk is monotone non-increasing in the abstention threshold ``λ`` (abstain
    on more uncertain voxels ⇒ lower risk on the retained set). RCPS selects

    .. math:: \hat\lambda = \inf\{\lambda : \mathrm{UCB}(\hat R(\lambda')) \le \alpha\ \forall \lambda' \ge \lambda\}.

    Args:
        risk_curve: ``[(lambda, empirical_risk_at_lambda), ...]`` (any order).
        alpha: risk budget.
        delta: confidence parameter.
        n: cohort size used for the Hoeffding gap.
        loss_bound: risk range.

    Returns:
        ``hat_lambda`` (float) or ``inf`` if no threshold controls the risk.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
    curve = sorted(risk_curve, key=lambda t: t[0])
    # Walk from the largest lambda downward; the RCPS set is the suffix whose every
    # member satisfies the UCB. Return the smallest lambda that still holds.
    chosen = float("inf")
    for lam, risk in reversed(curve):
        ucb = hoeffding_risk_upper_bound(risk, n, delta, loss_bound)
        if ucb <= alpha:
            chosen = lam
        else:
            break
    return chosen


@dataclass(frozen=True)
class DiceRiskReport:
    """Cohort-level Dice-risk RCPS report (serialised to JSON)."""

    alpha: float
    delta: float
    n: int
    empirical_risk: float
    certified_risk_upper_bound: float
    is_certified: bool
    rcps_threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DiceRiskCertificate:
    """Hoeffding-UCB risk-control gate for the expected Dice-risk (B-1.7).

    Stateless: call :meth:`certify` with the cohort's empirical mean Dice-risk and
    size. ``is_certified`` is ``True`` iff the UCB clears the budget ``alpha``.
    """

    def __init__(self, alpha: float = 0.1, delta: float = 0.05, loss_bound: float = 1.0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {alpha}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        if loss_bound <= 0.0:
            raise ValueError(f"loss_bound must be > 0; got {loss_bound}")
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.loss_bound = float(loss_bound)

    def certify(
        self,
        empirical_risk: float,
        n: int,
        rcps_threshold_value: float | None = None,
    ) -> DiceRiskReport:
        ucb = hoeffding_risk_upper_bound(empirical_risk, n, self.delta, self.loss_bound)
        return DiceRiskReport(
            alpha=self.alpha,
            delta=self.delta,
            n=int(n),
            empirical_risk=float(empirical_risk),
            certified_risk_upper_bound=ucb,
            is_certified=ucb <= self.alpha,
            rcps_threshold=rcps_threshold_value,
        )


__all__ = [
    "DiceRiskCertificate",
    "DiceRiskReport",
    "hoeffding_risk_upper_bound",
    "rcps_threshold",
]
