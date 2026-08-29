r"""Cross-site PAC-Bayes generalisation certificate (ULF roadmap PR-24 / R5).

Implements the McAllester PAC-Bayes bound from
``TODO/backlog_ulf_radiologist_acceptance_roadmap.md`` §R5.

Theorem R5 (PAC-Bayes federated generalisation, McAllester 1999)
----------------------------------------------------------------
For a posterior :math:`Q` over network parameters trained on
:math:`K` sites with :math:`n = \sum_k n_k` total samples, and a
prior :math:`P` (typically the foundation-model initialisation),
with probability :math:`\ge 1 - \delta` over the data draw:

.. math::

    \mathbb{E}_{\theta \sim Q}\!\Bigl[\tfrac{1}{K}\sum_k L_k(\theta)\Bigr]
    \;\le\;
    \mathbb{E}_{\theta \sim Q}\!\Bigl[\tfrac{1}{K}\sum_k \hat L_k(\theta)\Bigr]
    + \sqrt{\frac{\mathrm{KL}(Q\,\|\,P) + \ln(2\sqrt n / \delta)}{2(n-1)}}.

Extension to held-out sites
---------------------------
For a new site at total-variation distance :math:`\le \tau` from the
convex hull of the training sites, add :math:`\tau \cdot M` to the
right-hand side, where :math:`M` is the per-sample loss bound.

Non-vacuous bound condition
---------------------------
The bound is only operationally useful when
:math:`\mathrm{KL}(Q\|P)` is small — the framework satisfies this by
initialising :math:`Q` from a pretrained foundation model and
penalising divergence during federated rounds.

References
----------
McAllester D. A. *PAC-Bayesian Model Averaging*. COLT 1999.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def mcallester_complexity_term(
    kl_q_p: float,
    n: int,
    delta: float,
) -> float:
    r"""The McAllester complexity term :math:`\sqrt{(KL + \ln(2\sqrt n/\delta))/(2(n-1))}`.

    Args:
        kl_q_p: Kullback-Leibler divergence ``KL(Q ‖ P)``.
        n: Total sample count across all sites.
        delta: Confidence parameter ``δ ∈ (0, 1)``.

    Raises:
        ValueError: invalid arguments.
    """
    if kl_q_p < 0:
        raise ValueError(f"kl_q_p must be ≥ 0; got {kl_q_p}")
    if n < 2:
        raise ValueError(f"n must be ≥ 2; got {n}")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta}")
    log_term = math.log(2.0 * math.sqrt(n) / delta)
    return math.sqrt((kl_q_p + log_term) / (2.0 * (n - 1)))


def pac_bayes_bound(
    empirical_risk: float,
    kl_q_p: float,
    n: int,
    delta: float,
    tv_tolerance: float = 0.0,
    loss_bound_M: float = 1.0,
) -> float:
    r"""Combine the empirical risk + McAllester slack + optional TV term.

    Args:
        empirical_risk: ``E_Q[(1/K) Σ_k hat L_k]``.
        kl_q_p: ``KL(Q ‖ P)``.
        n: Total sample count.
        delta: Confidence parameter.
        tv_tolerance: Total-variation distance tolerance ``τ`` for a
            held-out site. ``0`` (default) → train-distribution bound.
        loss_bound_M: Per-sample loss range :math:`M` used by the
            ``τ·M`` extension term.

    Returns:
        Certified upper bound on the cross-site generalisation risk.

    Raises:
        ValueError: invalid arguments.
    """
    if empirical_risk < 0:
        raise ValueError(f"empirical_risk must be ≥ 0; got {empirical_risk}")
    if tv_tolerance < 0:
        raise ValueError(f"tv_tolerance must be ≥ 0; got {tv_tolerance}")
    if loss_bound_M <= 0:
        raise ValueError(f"loss_bound_M must be > 0; got {loss_bound_M}")
    slack = mcallester_complexity_term(kl_q_p, n, delta)
    return float(empirical_risk + slack + tv_tolerance * loss_bound_M)


@dataclass(frozen=True)
class PACBayesReport:
    """Per-experiment PAC-Bayes certificate.

    Attributes:
        empirical_risk: ``E_Q[(1/K) Σ_k hat L_k]``.
        kl_q_p: ``KL(Q ‖ P)``.
        n: Total sample count.
        delta: Confidence parameter.
        tv_tolerance: ``τ`` for the held-out-site extension.
        loss_bound_M: Per-sample loss range used by the τ·M term.
        complexity_term: ``√((KL + ln(2√n/δ)) / (2(n-1)))``.
        bound: The certified upper bound (final number).
        is_non_vacuous: ``True`` iff the bound is < ``loss_bound_M``
            (operationally meaningful — otherwise it gives the trivial
            "loss ≤ M" guarantee).
    """

    empirical_risk: float
    kl_q_p: float
    n: int
    delta: float
    tv_tolerance: float
    loss_bound_M: float
    complexity_term: float
    bound: float
    is_non_vacuous: bool

    def to_dict(self) -> dict:
        """JSON-friendly serialisation for the regulatory package."""
        return asdict(self)


class PACBayesCertificate:
    """Audit-style wrapper that runs the bound + non-vacuous check.

    Args:
        delta: Confidence parameter ``δ ∈ (0, 1)``.
        loss_bound_M: Per-sample loss range :math:`M`. The bound is
            "non-vacuous" iff it ends up below this value — otherwise
            the trivial guarantee ``loss ≤ M`` already holds.
        tv_tolerance: Total-variation tolerance for the held-out-site
            extension. Set to 0 for the train-distribution bound.

    Stateless: call :py:meth:`certify` with empirical risk + KL.
    """

    def __init__(
        self,
        delta: float = 0.05,
        loss_bound_M: float = 1.0,
        tv_tolerance: float = 0.0,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {delta}")
        if loss_bound_M <= 0:
            raise ValueError(f"loss_bound_M must be > 0; got {loss_bound_M}")
        if tv_tolerance < 0:
            raise ValueError(f"tv_tolerance must be ≥ 0; got {tv_tolerance}")
        self.delta = float(delta)
        self.loss_bound_M = float(loss_bound_M)
        self.tv_tolerance = float(tv_tolerance)

    def certify(
        self,
        empirical_risk: float,
        kl_q_p: float,
        n: int,
    ) -> PACBayesReport:
        """Run the bound + non-vacuous check on the supplied inputs."""
        slack = mcallester_complexity_term(kl_q_p, n, self.delta)
        bound = pac_bayes_bound(
            empirical_risk=empirical_risk,
            kl_q_p=kl_q_p,
            n=n,
            delta=self.delta,
            tv_tolerance=self.tv_tolerance,
            loss_bound_M=self.loss_bound_M,
        )
        return PACBayesReport(
            empirical_risk=float(empirical_risk),
            kl_q_p=float(kl_q_p),
            n=int(n),
            delta=self.delta,
            tv_tolerance=self.tv_tolerance,
            loss_bound_M=self.loss_bound_M,
            complexity_term=slack,
            bound=bound,
            is_non_vacuous=bound < self.loss_bound_M,
        )


__all__ = [
    "PACBayesCertificate",
    "PACBayesReport",
    "mcallester_complexity_term",
    "pac_bayes_bound",
]
