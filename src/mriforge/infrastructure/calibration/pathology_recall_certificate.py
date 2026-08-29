r"""Pathology Recall Certificate (PRC) — ULF-PR-23 / R4.

Implements the Hoeffding lower-bound certificate from
``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` §R4.

Theorem R4 (Hoeffding lower bound on recall)
--------------------------------------------
For a binary detector with empirical recall :math:`\hat r` evaluated
on :math:`n` independent positive samples, with probability
:math:`\ge 1 - \delta` over the test draw:

.. math::

    r \;\ge\; \hat r - \sqrt{\frac{\ln(1/\delta)}{2 n}}.

The bound is a direct application of Hoeffding's inequality (1963)
to the indicator-recall random variable.

Operational consequences (reproduced from the plan)
---------------------------------------------------
The Hoeffding gap :math:`\sqrt{\ln(1/\delta) / (2n)}` is the price
clinical claims pay for finite test sets. At :math:`\delta = 0.05,
n = 200`, the gap is ≈ 0.087: to certify "recall ≥ 0.90", the
empirical recall must be ≥ 0.987.

This module provides three pieces:

- :func:`hoeffding_recall_lower_bound` — the closed-form bound
  itself, exported separately for re-use by other certificates.
- :class:`PathologyRecallCertificate` — wraps the audit gate
  semantics (``is_certified`` / ``required_empirical_recall``).
- :class:`PathologyRecallReport` — the per-class result that the
  reporting layer serialises to ``pathology_certificates.json``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def hoeffding_recall_lower_bound(
    empirical_recall: float,
    n_positives: int,
    delta: float,
) -> float:
    r"""Return :math:`\hat r - \sqrt{\ln(1/\delta) / (2 n)}`, clamped to ``[0, 1]``.

    Args:
        empirical_recall: ``hat r ∈ [0, 1]``.
        n_positives: Number of positive examples in the test set.
        delta: Confidence parameter ``δ ∈ (0, 1)``.

    Raises:
        ValueError: invalid arguments.
    """
    if not 0.0 <= empirical_recall <= 1.0:
        raise ValueError(f"empirical_recall must lie in [0, 1]; got {empirical_recall}")
    if n_positives <= 0:
        raise ValueError(f"n_positives must be > 0; got {n_positives}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    gap = math.sqrt(math.log(1.0 / delta) / (2.0 * n_positives))
    return max(0.0, empirical_recall - gap)


def required_empirical_recall(
    target_recall: float,
    n_positives: int,
    delta: float,
) -> float:
    r"""Empirical-recall threshold needed to certify ``r ≥ target``.

    Inverts the Hoeffding bound: ``hat r >= target + gap`` certifies
    ``r >= target`` with probability ``1 - δ``.

    Returns ``inf`` (not a finite ``hat r``) when no value of
    ``hat r`` could certify the target — i.e. when ``target + gap > 1``.
    """
    if not 0.0 <= target_recall <= 1.0:
        raise ValueError(f"target_recall must lie in [0, 1]; got {target_recall}")
    if n_positives <= 0:
        raise ValueError(f"n_positives must be > 0; got {n_positives}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    gap = math.sqrt(math.log(1.0 / delta) / (2.0 * n_positives))
    needed = target_recall + gap
    return needed if needed <= 1.0 else float("inf")


@dataclass(frozen=True)
class PathologyRecallReport:
    """Per-class certificate emitted by :class:`PathologyRecallCertificate`.

    Serialises to JSON for ``reports/<exp>/pathology_certificates.json``.

    Attributes:
        pathology_class: Identifier (e.g. ``"infarct"``, ``"meningioma"``).
        empirical_recall: Observed recall on the test set.
        n_positives: Positive-class test count.
        delta: Confidence parameter.
        target_recall: Target recall the audit gate evaluates against.
        certified_recall_lower_bound: ``hat r - gap`` (Hoeffding bound).
        is_certified: ``True`` iff the bound clears ``target_recall``.
    """

    pathology_class: str
    empirical_recall: float
    n_positives: int
    delta: float
    target_recall: float
    certified_recall_lower_bound: float
    is_certified: bool

    def to_dict(self) -> dict:
        """JSON-friendly serialisation."""
        return asdict(self)


class PathologyRecallCertificate:
    """Run the Hoeffding-bound audit for a single pathology class.

    Args:
        pathology_class: Short identifier — used to key the output dict.
        target_recall: ``r_target ∈ [0, 1]`` the bound must clear.
        delta: Confidence parameter ``δ ∈ (0, 1)``.

    The certificate is *stateless*: call :py:meth:`certify` with the
    empirical recall + sample count to get the report.
    """

    def __init__(
        self,
        pathology_class: str,
        target_recall: float,
        delta: float = 0.05,
    ) -> None:
        if not pathology_class:
            raise ValueError("pathology_class must be a non-empty string")
        if not 0.0 <= target_recall <= 1.0:
            raise ValueError(f"target_recall must lie in [0, 1]; got {target_recall}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        self.pathology_class = str(pathology_class)
        self.target_recall = float(target_recall)
        self.delta = float(delta)

    def certify(
        self,
        empirical_recall: float,
        n_positives: int,
    ) -> PathologyRecallReport:
        """Compute the lower bound and check the certification gate."""
        bound = hoeffding_recall_lower_bound(empirical_recall, n_positives, self.delta)
        return PathologyRecallReport(
            pathology_class=self.pathology_class,
            empirical_recall=float(empirical_recall),
            n_positives=int(n_positives),
            delta=self.delta,
            target_recall=self.target_recall,
            certified_recall_lower_bound=bound,
            is_certified=bound >= self.target_recall,
        )

    def required_empirical_recall(self, n_positives: int) -> float:
        """Empirical recall needed to certify this certificate's target."""
        return required_empirical_recall(self.target_recall, n_positives, self.delta)


__all__ = [
    "PathologyRecallCertificate",
    "PathologyRecallReport",
    "hoeffding_recall_lower_bound",
    "required_empirical_recall",
]
