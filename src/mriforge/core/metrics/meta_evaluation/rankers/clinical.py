"""Gen-4 clinical / causal rankers — DCMS, ACSS, HiBSL.

These three rankers close the loop between physics-anchored severity θ and
*external* clinical ground truth — a frozen downstream task network (DCMS) or
radiologist Likert observations (ACSS, HiBSL). They are therefore
**capability-gated**: on a dataset that carries neither (e.g. M4Raw 0.3T, which
has no task net and no Likert), :func:`~...registry.available_rankers` omits them
— no silent fallback, an explicit, auditable omission. They are nonetheless
**registered** on the unified registry (with ``requires_likert`` /
``requires_task_net`` set) for completeness and so the Tier-1 audit can report
the unmet dependency by name.

This module is the **single source of truth** for the mediation / sufficiency /
calibration math (it moved inward from ``scripts/sim2rank`` — pitfall #12 / #13).
The legacy ``scripts/sim2rank/{causal_mediation,sufficiency,hibsl}.py`` modules
now *re-export* the moved functions (scripts may import inward) while keeping
their own scripts-registry ``@register_ranker`` class wrappers, so
``run_all_generations.py`` and ``tests/unit/test_sim2rank_tier2_rankers.py``
keep working unchanged.

The three :class:`BaseRanker` wrappers below take the external data at
``__init__`` (injected by the orchestrator or by ``from_registry(ranker_kwargs=…)``)
and :meth:`rank` **raises** if the data is ``None`` — never degrading to garbage.
The registry's ``data_kwargs`` declaration is what lets the capability gate verify
the data was actually *supplied*, not merely *declared* (issue #258).

What these three actually implement (be precise — issue #255d)
--------------------------------------------------------------
* **DCMS** is a *parametric* mediation estimator: per-anatomy-cell linear-in-θ
  mediator + linear-with-interaction outcome model, combined in the
  VanderWeele/Baron-Kenny closed form. It is **not** the Imai et al. nonparametric
  simulation estimator, and it does **not** test sequential ignorability — that is
  assumed. What it *does* now do is report ρ̃ (the confounding that would nullify
  the finding) and a positivity/overlap diagnostic, so the assumption is at least
  visible.
* **ACSS** is the Shah-Peters GCM on cross-fitted residuals. It ranks on the
  bounded residual partial correlation (an effect size) and reserves BH-FDR for
  the pass/fail decision, plus a TOST equivalence test that supplies *positive*
  evidence for sufficiency rather than a bare failure to reject.
* **HiBSL** is a *severity-anchored empirical-Bayes* calibration: image-level
  latent quality partially pooled toward a severity-regression prior, with
  per-rater ordered probits fit by maximum likelihood. There is **no MCMC**, no
  posterior, no R-hat/ESS and no pooling across raters; ``use_numpyro=True``
  raises rather than pretending otherwise.

References
----------
* Pearl, "Direct and indirect effects," UAI 2001.
* Robins & Greenland, "Identifiability and exchangeability for direct and
  indirect effects," Epidemiology 3(2), 1992.
* Baron & Kenny, "The moderator-mediator variable distinction," JPSP 51(6), 1986.
* VanderWeele, "A three-way decomposition of a total effect into direct, indirect
  and interactive effects," Epidemiology 24(2), 2013. — the decomposition actually
  implemented (NIE + NDE + mediated interaction).
* Imai, Keele & Yamamoto, "Identification, inference and sensitivity analysis for
  causal mediation effects," Statist. Sci. 25(1), 2010. — the source of the ρ̃
  sensitivity parameter only; the *estimator* here is not theirs.
* Shah & Peters, "The hardness of conditional independence testing and the
  generalised covariance measure," Ann. Statist. 48(3), 2020.
* Benjamini & Hochberg, "Controlling the false discovery rate," JRSS B 57(1), 1995.
* Lakens, "Equivalence tests," Soc. Psychol. Personal. Sci. 8(4), 2017. — TOST.
* Albert & Chib, "Bayesian analysis of binary and polychotomous response data,"
  JASA 88(422), 1993.
* Mason et al., "Comparison of objective image quality metrics to expert
  radiologists' scoring," IEEE TMI 39(4), 2020.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker

logger = logging.getLogger(__name__)


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


# ══════════════════════════════════════════════════════════════════════
# DCMS — Differentiable Causal Mediation Score
# (moved verbatim from scripts/sim2rank/causal_mediation.py)
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MediationData:
    """Observational data for the mediation analysis.

    Attributes:
        severity: ``(N,)`` physics-anchored severity θ ∈ [0, 1].
        metric_values: ``(N, M)`` per-image, per-metric values.
        task_performance: ``(N,)`` frozen-task performance P (Dice,
            AUROC, …) per image. *Higher is better* by convention.
        anatomy: ``(N,)`` integer anatomy/contrast cell or None.
    """

    severity: np.ndarray
    metric_values: np.ndarray
    task_performance: np.ndarray
    anatomy: np.ndarray | None = None


@dataclass(frozen=True)
class DCMSResult:
    """Per-metric DCMS output.

    Attributes:
        mediation_fraction: ``(M,)`` **the ranking score**, bounded in ``[0, 1]``:
            ``|NIE| / (|NIE| + |NDE| + |INT|)``. See :func:`compute_dcms` for why
            the textbook proportion-mediated cannot be ranked on.
        proportion_mediated: ``(M,)`` ``NIE / TE`` — reported for interpretability
            only. **Unbounded** (it explodes to ±∞ under suppression, when
            ``NIE`` and ``NDE`` nearly cancel), and ``NaN`` where ``|TE| ≈ 0``.
        nie: ``(M,)`` natural indirect effect (through the metric).
        nde: ``(M,)`` natural direct effect (bypassing the metric).
        interaction: ``(M,)`` mediated-interaction term ``TE - NIE - NDE``. Non-
            zero exactly when the outcome model's ``θ·m`` coefficient is non-zero.
        total_effect: ``(M,)`` the **directly evaluated** total effect
            ``E[P(θ₁, m(θ₁)) - P(θ₀, m(θ₀))]``. It is *not* ``NIE + NDE``.
        rho_zero: ``(M,)`` sensitivity parameter ρ̃ — the mediator/outcome error
            correlation at which the estimated NIE would vanish. Identification is
            *assumed*, never tested; ``|rho_zero|`` says how much unmeasured
            confounding it would take to explain the mediation away.
        positivity: Per-anatomy-cell overlap diagnostics (cell size, number of
            distinct severities, θ range). Mediation is not identified in a cell
            with fewer than two distinct severities.
        n_pairs: Number of (θ₀, θ₁) contrasts actually sampled.
    """

    mediation_fraction: np.ndarray  # (M,) bounded [0, 1] — the ranking score
    proportion_mediated: np.ndarray  # (M,) NIE / TE — unbounded, report-only
    nie: np.ndarray  # (M,) natural indirect effect
    nde: np.ndarray  # (M,) natural direct effect
    interaction: np.ndarray  # (M,) mediated interaction (TE - NIE - NDE)
    total_effect: np.ndarray  # (M,) directly evaluated TE
    rho_zero: np.ndarray  # (M,) Imai ρ̃ sensitivity parameter
    positivity: dict[str, object]
    n_pairs: int


def _sample_pairs(
    severity: np.ndarray,
    anatomy: np.ndarray | None,
    *,
    n_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample (low-severity, high-severity) index pairs, anatomy-blocked.

    Within each anatomy cell, pair indices i, j with θ_i < θ_j. This
    matches the sequential-ignorability assumption: severity is
    randomised within anatomy by the simulator's design.
    """
    rng = np.random.default_rng(seed)
    N = severity.shape[0]
    if anatomy is None:
        anatomy = np.zeros(N, dtype=np.int64)

    pairs: list[tuple[int, int]] = []
    for a_cell in np.unique(anatomy):
        in_cell = np.where(anatomy == a_cell)[0]
        if in_cell.size < 2:
            continue
        for _ in range(n_pairs // max(len(np.unique(anatomy)), 1)):
            i, j = rng.choice(in_cell, size=2, replace=False)
            if severity[i] > severity[j]:
                i, j = j, i
            if severity[i] == severity[j]:
                continue
            pairs.append((int(i), int(j)))
    return pairs


def _fit_mediator_model(
    severity: np.ndarray,
    anatomy: np.ndarray | None,
    metric: np.ndarray,
) -> Callable[[float, int], float]:
    """Fit a simple mediator model m_j(θ, A).

    Uses a per-anatomy-cell linear regression in θ. The Imai et al.
    estimator only requires the conditional distribution
    p(m_j | θ, A), not its functional form — a Gaussian residual
    around the fitted mean is sufficient for the simulation step.

    NaN-robust: drops any (θ, m) pair where either value is non-
    finite. Necessary because some metrics in the registry return
    NaN on certain degradations (e.g. divide-by-zero in cleanly-
    normalised inputs), and lstsq / polyfit raise on NaN inputs.
    """
    if anatomy is None:
        anatomy = np.zeros_like(severity, dtype=np.int64)

    cells = np.unique(anatomy)
    coefs: dict[int, tuple[float, float, float]] = {}
    for c in cells:
        sel = anatomy == c
        x = severity[sel]
        y = metric[sel]
        # Drop NaNs before polyfit.
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 2 or np.std(x) < 1e-12:
            coefs[int(c)] = (float(np.mean(y)) if y.size else 0.0, 0.0, 1.0)
            continue
        a, b = np.polyfit(x, y, 1)
        resid = y - (a * x + b)
        coefs[int(c)] = (float(b), float(a), float(np.std(resid) + 1e-12))

    def _predict(theta: float, a_cell: int) -> float:
        intercept, slope, _sd = coefs.get(a_cell, (0.0, 0.0, 1.0))
        return intercept + slope * theta

    return _predict


def _fit_outcome_model(
    severity: np.ndarray,
    metric: np.ndarray,
    anatomy: np.ndarray | None,
    task_perf: np.ndarray,
) -> Callable[[float, float, int], float]:
    """Fit a simple outcome model E[P | θ, m_j, A] = a + bθ + cm + dθ·m.

    A 4-parameter ordinary-least-squares fit per anatomy cell is
    enough to expose the direct and indirect paths. Higher-fidelity
    fits (gradient-boosted trees) are a v2 item.
    """
    if anatomy is None:
        anatomy = np.zeros_like(severity, dtype=np.int64)

    cells = np.unique(anatomy)
    coefs: dict[int, np.ndarray] = {}
    for c in cells:
        sel = anatomy == c
        if sel.sum() < 4:
            valid_y = task_perf[sel]
            valid_y = valid_y[np.isfinite(valid_y)]
            coefs[int(c)] = np.array(
                [
                    float(np.mean(valid_y)) if valid_y.size else 0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            continue
        sev_c = severity[sel]
        met_c = metric[sel]
        y = task_perf[sel]
        # Drop rows with any non-finite covariate or outcome — lstsq
        # raises ``SVD did not converge`` on NaN inputs.
        mask = np.isfinite(sev_c) & np.isfinite(met_c) & np.isfinite(y)
        if int(mask.sum()) < 4:
            valid_y = y[mask]
            coefs[int(c)] = np.array(
                [
                    float(np.mean(valid_y)) if valid_y.size else 0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            continue
        X = np.stack(
            [
                np.ones(int(mask.sum())),
                sev_c[mask],
                met_c[mask],
                sev_c[mask] * met_c[mask],
            ],
            axis=1,
        )
        y = y[mask]
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            # Pseudo-inverse fallback for pathological designs.
            coef = np.linalg.pinv(X) @ y
        coefs[int(c)] = np.asarray(coef, dtype=np.float64)

    def _predict(theta: float, m: float, a_cell: int) -> float:
        c = coefs.get(a_cell, np.zeros(4))
        return float(c[0] + c[1] * theta + c[2] * m + c[3] * theta * m)

    return _predict


def _rho_sensitivity(
    severity: np.ndarray,
    anatomy: np.ndarray,
    metric: np.ndarray,
    task_perf: np.ndarray,
) -> float:
    r"""Imai-Keele-Yamamoto ρ̃ — the confounding that would nullify the NIE.

    For the linear structural model
    :math:`m = \alpha_2 + \beta_2\theta + \varepsilon_2`,
    :math:`P = \alpha_1 + \beta_1\theta + \varepsilon_1`, the ACME as a function of
    the mediator/outcome error correlation :math:`\rho = \mathrm{corr}(\varepsilon_2,
    \varepsilon_3)` vanishes exactly at :math:`\rho = \tilde\rho \equiv
    \mathrm{corr}(\varepsilon_1, \varepsilon_2)` (Imai et al. 2010, §4.1). So ρ̃ *is*
    the sensitivity parameter: an unmeasured confounder inducing an error
    correlation of ρ̃ would drive the estimated indirect effect to zero.

    Reported, not enforced: sequential ignorability is **assumed**, and this
    quantifies how fragile that assumption makes the conclusion. Residuals are
    taken within anatomy cell and pooled. The closed form is exact for the
    no-interaction linear SEM and is a first-order approximation for the
    interaction model actually fitted here.

    Returns:
        ρ̃ ∈ ``[-1, 1]``, or ``NaN`` when either residual has no variance.
    """
    resid_p: list[np.ndarray] = []
    resid_m: list[np.ndarray] = []
    for c in np.unique(anatomy):
        sel = anatomy == c
        x, m_c, p_c = severity[sel], metric[sel], task_perf[sel]
        mask = np.isfinite(x) & np.isfinite(m_c) & np.isfinite(p_c)
        if int(mask.sum()) < 3 or np.std(x[mask]) < 1e-12:
            continue
        x, m_c, p_c = x[mask], m_c[mask], p_c[mask]
        am, bm = np.polyfit(x, m_c, 1)
        ap, bp = np.polyfit(x, p_c, 1)
        resid_m.append(m_c - (am * x + bm))
        resid_p.append(p_c - (ap * x + bp))
    if not resid_m:
        return float("nan")
    e2 = np.concatenate(resid_m)
    e1 = np.concatenate(resid_p)
    if np.std(e1) < 1e-12 or np.std(e2) < 1e-12:
        return float("nan")
    return float(np.corrcoef(e1, e2)[0, 1])


def _positivity_report(severity: np.ndarray, anatomy: np.ndarray) -> dict[str, object]:
    """Overlap / positivity diagnostics per anatomy cell.

    Mediation is not identified in a cell that carries a single severity value
    (no contrast to average over), and is estimated on thin air in a cell of two
    or three images. The estimator silently skipped such cells; now they are
    reported so a caller can see how much of the pool actually contributed.
    """
    cells: dict[str, dict[str, float]] = {}
    degenerate: list[str] = []
    for c in np.unique(anatomy):
        sel = anatomy == c
        theta = severity[sel]
        theta = theta[np.isfinite(theta)]
        n_distinct = int(np.unique(theta).size)
        cells[str(int(c))] = {
            "n": int(theta.size),
            "n_distinct_severity": n_distinct,
            "theta_min": float(theta.min()) if theta.size else float("nan"),
            "theta_max": float(theta.max()) if theta.size else float("nan"),
        }
        if n_distinct < 2:
            degenerate.append(str(int(c)))
    return {
        "cells": cells,
        "degenerate_cells": degenerate,
        "positivity_ok": not degenerate,
    }


def compute_dcms(
    data: MediationData,
    *,
    n_pairs: int = 256,
    seed: int = 0,
) -> DCMSResult:
    r"""Causal mediation decomposition of severity → task performance per metric.

    What this actually is (issue #255d)
    -----------------------------------
    A **parametric** mediation estimator: a per-anatomy-cell linear-in-θ mediator
    model :math:`m_j(\theta) = b_0 + b_1\theta` and a linear-with-interaction
    outcome model :math:`P(\theta, m) = c_0 + c_1\theta + c_2 m + c_3\theta m`,
    combined in the VanderWeele/Baron-Kenny closed form. It is **not** the Imai et
    al. nonparametric simulation estimator (there is no posterior draw and no
    mediator simulation), and earlier docstrings claiming so were wrong.
    Identification (sequential ignorability, no unmeasured mediator-outcome
    confounder) is **assumed, not tested**; :attr:`DCMSResult.rho_zero` reports
    how much confounding would be needed to nullify the finding, and
    :attr:`DCMSResult.positivity` reports the overlap the estimate rests on.

    The decomposition closes (issue #255a)
    --------------------------------------
    With a fitted ``θ·m`` interaction the total effect is **not** ``NIE + NDE``:

    .. math::

        \mathrm{TE} = \underbrace{(c_2 + c_3\theta_0)\,\Delta m}_{\mathrm{NIE}}
                    + \underbrace{(c_1 + c_3 m(\theta_0))\,\Delta\theta}_{\mathrm{NDE}}
                    + \underbrace{c_3\,\Delta\theta\,\Delta m}_{\mathrm{INT}}

    The old code fitted ``c_3`` and then threw the third term away, reporting
    ``TE := NIE + NDE`` — a 29.6% error on a synthetic sweep with a real
    interaction. TE is now evaluated **directly** as
    :math:`\mathbb{E}[P(\theta_1, m(\theta_1)) - P(\theta_0, m(\theta_0))]` and the
    mediated interaction is recovered as the residual, so the three parts sum to
    TE by construction.

    Why the score is not the proportion-mediated (issue #255b)
    ---------------------------------------------------------
    ``PM = NIE / TE`` is **unbounded**. Under *suppression* — NIE and NDE of
    opposite sign, nearly cancelling — TE → 0 and PM explodes (a measured
    ``NIE=+0.344, NDE=-0.338`` gives ``PM=+54.3``), so a metric whose effects
    merely cancel outranks an honestly fully-mediating metric at ``PM=1.0``. The
    ranking score is therefore the bounded

    .. math::

        \mathrm{MF}_j = \frac{|\mathrm{NIE}_j|}
                             {|\mathrm{NIE}_j| + |\mathrm{NDE}_j| + |\mathrm{INT}_j|}
              \in [0, 1],

    which is 1 for a purely-indirect effect, 0 for a purely-direct one, and cannot
    be inflated by cancellation. PM is still reported, for interpretation only.

    Crashed metrics stay NaN (issue #255c)
    --------------------------------------
    A metric that crashed on its whole cohort (all-NaN column) has no estimable
    effect. It scores ``NaN`` and :func:`ranking_guards.ranks_from_scores` puts it
    LAST. The old ``np.nan_to_num(pm, nan=0.0)`` laundered it into a finite
    ``0.0``, ranking it *above* an honest metric at ``PM = -0.999`` and defeating
    the crash→NaN→ranks-last contract that ``base.py`` documents.

    Args:
        data: Bundled severity, metric_values, task_performance, anatomy.
        n_pairs: Number of (θ_0, θ_1) contrasts per anatomy cell.
        seed: RNG seed.
    """
    M = data.metric_values.shape[1]
    anatomy = (
        data.anatomy if data.anatomy is not None else np.zeros_like(data.severity, dtype=np.int64)
    )
    positivity = _positivity_report(data.severity, anatomy)

    pairs = _sample_pairs(
        data.severity,
        data.anatomy,
        n_pairs=n_pairs,
        seed=seed,
    )
    if not pairs:
        nan = np.full(M, np.nan)
        return DCMSResult(
            mediation_fraction=nan.copy(),
            proportion_mediated=nan.copy(),
            nie=nan.copy(),
            nde=nan.copy(),
            interaction=nan.copy(),
            total_effect=nan.copy(),
            rho_zero=nan.copy(),
            positivity=positivity,
            n_pairs=0,
        )

    nie_per_metric = np.full(M, np.nan, dtype=np.float64)
    nde_per_metric = np.full(M, np.nan, dtype=np.float64)
    te_per_metric = np.full(M, np.nan, dtype=np.float64)
    rho_per_metric = np.full(M, np.nan, dtype=np.float64)

    for j in range(M):
        metric_j = data.metric_values[:, j]
        # A metric with no finite value anywhere carries no signal — the fitted
        # "model" would be the all-zero fallback and every effect would come out
        # a spurious 0.0. Leave the whole row NaN (crash → ranks LAST).
        if not np.any(np.isfinite(metric_j)):
            continue
        try:
            mediator = _fit_mediator_model(data.severity, anatomy, metric_j)
            outcome = _fit_outcome_model(
                data.severity,
                metric_j,
                anatomy,
                data.task_performance,
            )
        except (np.linalg.LinAlgError, ValueError):
            # Per-metric regression failed — leave the row NaN, never 0.0.
            continue

        nie_sum = 0.0
        nde_sum = 0.0
        te_sum = 0.0
        for i, k in pairs:
            theta_0 = float(data.severity[i])
            theta_1 = float(data.severity[k])
            a_cell = int(anatomy[i])

            m_at_0 = mediator(theta_0, a_cell)
            m_at_1 = mediator(theta_1, a_cell)

            p_baseline = outcome(theta_0, m_at_0, a_cell)

            # NIE: swap m_j(θ_0) → m_j(θ_1) with θ held at θ_0.
            nie_sum += outcome(theta_0, m_at_1, a_cell) - p_baseline
            # NDE: swap θ_0 → θ_1 with m_j held at m_j(θ_0).
            nde_sum += outcome(theta_1, m_at_0, a_cell) - p_baseline
            # TE: move BOTH — evaluated directly, never as NIE + NDE.
            te_sum += outcome(theta_1, m_at_1, a_cell) - p_baseline

        n = len(pairs)
        nie_per_metric[j] = nie_sum / n
        nde_per_metric[j] = nde_sum / n
        te_per_metric[j] = te_sum / n
        rho_per_metric[j] = _rho_sensitivity(
            data.severity, anatomy, metric_j, data.task_performance
        )

    # The mediated interaction is what the old TE := NIE + NDE silently discarded.
    interaction = te_per_metric - nie_per_metric - nde_per_metric

    # Report-only proportion-mediated, against the TRUE total effect. NaN where the
    # total effect is ~0 (the proportion is genuinely undefined) — and NOT laundered.
    with np.errstate(invalid="ignore", divide="ignore"):
        denom_te = np.where(np.abs(te_per_metric) > 1e-12, te_per_metric, np.nan)
        pm = nie_per_metric / denom_te

        # Bounded ranking score. A metric with no effect of any kind (all three
        # parts zero) has no evidence either way → NaN, not a spurious 0.
        scale = np.abs(nie_per_metric) + np.abs(nde_per_metric) + np.abs(interaction)
        denom_mf = np.where(scale > 1e-12, scale, np.nan)
        mediation_fraction = np.abs(nie_per_metric) / denom_mf

    return DCMSResult(
        mediation_fraction=mediation_fraction,
        proportion_mediated=pm,
        nie=nie_per_metric,
        nde=nde_per_metric,
        interaction=interaction,
        total_effect=te_per_metric,
        rho_zero=rho_per_metric,
        positivity=positivity,
        n_pairs=len(pairs),
    )


def baron_kenny_proportion_mediated(
    dP_dtheta: np.ndarray,
    dP_dmetric: np.ndarray,
    dmetric_dtheta: np.ndarray,
) -> np.ndarray:
    r"""Baron-Kenny linear-approximation PM.

    .. math::

        \mathrm{PM}_j \approx
            \frac{(\partial P / \partial m_j) \cdot (\partial m_j / \partial \theta)}
                 {\partial P / \partial \theta}

    All inputs are ``(M,)`` arrays of partial derivatives at a single
    pair (θ_0, θ_1). Returns NaN entries where the total effect is
    near zero (proportion is undefined).
    """
    nie = dP_dmetric * dmetric_dtheta
    nde = dP_dtheta - nie
    total = nde + nie
    with np.errstate(invalid="ignore", divide="ignore"):
        pm = np.where(np.abs(total) > 1e-12, nie / total, np.nan)
    return pm


# ══════════════════════════════════════════════════════════════════════
# ACSS — Anatomy-Conditional Sufficient Statistic
# (moved verbatim from scripts/sim2rank/sufficiency.py)
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SufficiencyData:
    """Inputs to the ACSS sufficiency test.

    Attributes:
        likert: ``(N,)`` per-image Likert (or any clinical-quality
            scalar). If multiple raters scored each image, aggregate
            to per-image mean Likert before passing in.
        severity: ``(N,)`` physics-anchored severity θ ∈ [0, 1].
        metric_values: ``(N, M)`` per-image, per-metric values.
        anatomy: ``(N,)`` integer anatomy/contrast cell or None.
    """

    likert: np.ndarray
    severity: np.ndarray
    metric_values: np.ndarray
    anatomy: np.ndarray | None = None


@dataclass(frozen=True)
class ACSSResult:
    """Per-metric ACSS output.

    Attributes:
        residual_correlation: ``(M,)`` **the ranking score's basis** — the partial
            correlation ``corr(L - Ê[L|m,A], θ - Ê[θ|m,A])`` ∈ ``[-1, 1]``. A
            *signed, bounded effect size*: 0 means the metric fully screens θ off
            from the Likert (sufficiency); ±1 means it screens off nothing.
        gcm_statistic: ``(M,)`` studentised GCM statistic ``T``.
        p_value: ``(M,)`` unadjusted p-value under the chosen null.
        p_value_bh: ``(M,)`` Benjamini-Hochberg adjusted p. Drives the FDR-
            controlled **decision** only — never the ordering (see
            :func:`compute_acss`).
        defensible: ``(M,)`` bool — the metric survives the FDR-controlled
            sufficiency screen (``p_bh > fdr_alpha``). A *failure to reject*, so
            it is as much a statement about N as about the metric.
        equivalence_p: ``(M,)`` TOST p-value for ``|partial corr| <
            equivalence_margin``. **This** is the positive evidence: small ⇒ the
            residual dependence is demonstrably negligible, not merely undetected.
        sufficient_equivalent: ``(M,)`` bool — TOST rejects non-equivalence at
            ``fdr_alpha``: positively certified as approximately sufficient.
        equivalence_margin: The margin the TOST was run against.
        null: Which null the p-values were computed under.
        n_samples: N.
    """

    residual_correlation: np.ndarray  # (M,) bounded signed effect size
    gcm_statistic: np.ndarray  # (M,) T_GCM
    p_value: np.ndarray  # (M,) unadjusted p-value under `null`
    p_value_bh: np.ndarray  # (M,) Benjamini-Hochberg adjusted p
    defensible: np.ndarray  # (M,) bool — adjusted p > alpha
    equivalence_p: np.ndarray  # (M,) TOST p for |r| < margin
    sufficient_equivalent: np.ndarray  # (M,) bool — TOST certifies sufficiency
    equivalence_margin: float
    null: str
    n_samples: int


def _cross_fit_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Cross-fitted out-of-fold regression predictions.

    Uses scikit-learn ``HistGradientBoostingRegressor`` when available
    (handles missing values and modest non-linearities), else falls
    back to ridge regression. The product-rate condition on the
    cross-fitted regression's L^2-error (Shah-Peters Th. 6) is
    satisfied by either choice at typical sample sizes.

    Args:
        X: ``(N, d)`` covariate matrix.
        y: ``(N,)`` regression target.
        n_folds: K-fold cross-fitting (default 5).
        seed: RNG seed for fold assignment.

    Returns:
        ``(N,)`` out-of-fold predictions.
    """
    from sklearn.model_selection import KFold

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as _Reg

        _kwargs = {"max_iter": 200, "max_depth": 6, "random_state": seed}
        _name = "HGBR"
    except ImportError:  # pragma: no cover
        from sklearn.linear_model import Ridge as _Reg

        _kwargs = {"alpha": 1.0, "random_state": seed}
        _name = "Ridge"

    preds = np.zeros_like(y, dtype=np.float64)
    kf = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        reg = _Reg(**_kwargs)
        reg.fit(X[train_idx], y[train_idx])
        preds[test_idx] = reg.predict(X[test_idx])
    logger.debug("ACSS: cross-fit with %s, K=%d, N=%d", _name, n_folds, len(y))
    return preds


#: Relative floor on the GCM's studentising scale, as a fraction of the natural
#: scale of the residual product ``e·f`` (issue #257c). The old ``max(σ², 1e-12)``
#: was an ABSOLUTE floor applied to a **scale-invariant** Wald ratio: multiplying
#: both residuals by ``c`` scales ``σ`` by ``c²`` and leaves ``T`` unchanged in
#: exact arithmetic, but once ``σ²`` fell under 1e-12 the floor took over and
#: ``T`` collapsed toward 0 — i.e. toward "sufficient". The bias always favours a
#: verdict of sufficiency, which is precisely the verdict ACSS exists to scrutinise.
_GCM_SIGMA_REL_EPS = 1e-8


def _gcm_statistic_and_pvalue(
    e_hat: np.ndarray,
    f_hat: np.ndarray,
) -> tuple[float, float]:
    """Compute T_GCM and the two-sided asymptotic p-value.

    ``T = √N · mean(ê f̂) / σ̂`` with ``σ̂`` the empirical SD of the per-sample
    product. The studentising scale is floored **relatively** — as a fraction of
    ``√(E[ê²]·E[f̂²])``, the natural magnitude of the product — so the statistic
    stays invariant to a rescaling of the residuals (issue #257c). Only a genuinely
    degenerate case (a residual that is identically zero) short-circuits, and it
    does so to ``T = 0, p = 1``: with no residual variation there is no evidence of
    conditional dependence.

    Returns:
        (T, p) where p = 2 (1 - Φ(|T|)).
    """
    from scipy.stats import norm

    N = e_hat.shape[0]
    products = e_hat * f_hat
    mean_prod = float(products.mean())
    sigma_sq = float(np.mean(products**2) - mean_prod**2)

    # Natural scale of the product — what a "small" sigma must be small RELATIVE to.
    scale = float(np.sqrt(np.mean(e_hat**2) * np.mean(f_hat**2)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 0.0, 1.0
    sigma = max(float(np.sqrt(max(sigma_sq, 0.0))), _GCM_SIGMA_REL_EPS * scale)

    T = float(np.sqrt(N) * mean_prod / sigma)
    p = 2.0 * (1.0 - float(norm.cdf(abs(T))))
    return T, p


def _residual_partial_correlation(e_hat: np.ndarray, f_hat: np.ndarray) -> float:
    """``corr(ê, f̂)`` ∈ ``[-1, 1]`` — the bounded ACSS sufficiency effect size.

    Zero ⇔ the metric screens severity off from the Likert (the sufficiency the
    test is looking for). This is what the ranking is built on: a p-value answers
    "could this be chance at THIS N", an effect size answers "how much residual
    dependence is left", and only the latter is a property of the metric.
    """
    sd_e = float(np.std(e_hat))
    sd_f = float(np.std(f_hat))
    if sd_e < 1e-30 or sd_f < 1e-30:
        return 0.0
    return float(np.clip(np.corrcoef(e_hat, f_hat)[0, 1], -1.0, 1.0))


def _tost_equivalence_pvalue(r: float, n: int, margin: float) -> float:
    """Two-one-sided-tests p for ``|corr| < margin``, via Fisher's z (issue #257b).

    ACSS's ``defensible`` flag is a **failure to reject** ``H₀: independence``, so
    it is a statement about power as much as about the metric: the same metric,
    same generative process, comes out "defensible" at N=40 and not at N=3000.
    TOST inverts the logic — it puts the *interesting* claim in the alternative:

    * ``H₀: |ρ| ≥ margin`` (the residual dependence is materially large)
    * ``H₁: |ρ| < margin`` (it is negligible — approximate sufficiency)

    A small TOST p is then positive evidence FOR sufficiency, and an underpowered
    study can no longer manufacture it (small N ⇒ wide CI ⇒ TOST fails).

    Returns:
        ``max`` of the two one-sided p-values; ``1.0`` when N is too small for the
        Fisher-z variance to exist.
    """
    from scipy.stats import norm

    if n <= 3 or margin <= 0.0:
        return 1.0
    z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    z_margin = float(np.arctanh(min(margin, 0.999999)))
    se = 1.0 / np.sqrt(n - 3)
    # Reject H0: z <= -z_margin  AND  H0: z >= +z_margin.
    p_lower = 1.0 - float(norm.cdf((z + z_margin) / se))
    p_upper = float(norm.cdf((z - z_margin) / se))
    return float(max(p_lower, p_upper))


def _gcm_permutation_pvalue(
    e_hat: np.ndarray,
    f_hat: np.ndarray,
    anatomy: np.ndarray | None,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Exact permutation null for the GCM — avoids the σ→0 Wald degeneracy (§5).

    The Wald statistic :math:`T=\\sqrt N\\,\\overline{ef}/\\hat\\sigma` has a vanishing
    denominator at *exact sufficiency* (``f→0 ⇒ σ²→0``) — precisely the favourable
    regime the test is meant to celebrate — so its Gaussian calibration breaks there.
    This permutation null sidesteps σ entirely: it permutes ``f`` *within anatomy
    cells* (respecting the conditional null :math:`L\\perp\\theta\\mid(m,A)`),
    recomputes :math:`\\overline{ef}`, and reads the two-sided ``+1``-corrected
    permutation rank as an exact p-value. The reported statistic is studentised by
    the *permutation* standard deviation, which stays well-defined as ``σ→0``.

    Returns:
        ``(T, p)`` — permutation-studentised effect and exact two-sided p-value.
    """
    rng = np.random.default_rng(seed)
    n = e_hat.shape[0]
    obs = float(np.mean(e_hat * f_hat))
    if anatomy is None:
        cells = [np.arange(n)]
    else:
        cells = [np.where(anatomy == c)[0] for c in np.unique(anatomy)]
    null = np.empty(n_perm, dtype=np.float64)
    for b in range(n_perm):
        f_perm = f_hat.copy()
        for idx in cells:
            f_perm[idx] = f_hat[idx][rng.permutation(idx.shape[0])]
        null[b] = float(np.mean(e_hat * f_perm))
    p = (1 + int(np.sum(np.abs(null) >= abs(obs)))) / (1 + n_perm)
    sd = float(np.std(null))
    T = obs / sd if sd > 1e-30 else 0.0
    return T, float(p)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up adjusted p-values."""
    M = p_values.shape[0]
    if M == 0:
        return p_values.copy()
    order = np.argsort(p_values)
    ranked = p_values[order]
    adj = ranked * M / np.arange(1, M + 1)
    # Enforce monotonicity from the largest rank down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def _one_hot(labels: np.ndarray) -> np.ndarray:
    """Simple one-hot encoder; drops the last column for identifiability."""
    cats = np.unique(labels)
    if len(cats) <= 1:
        return np.zeros((labels.shape[0], 0), dtype=np.float64)
    out = np.zeros((labels.shape[0], len(cats) - 1), dtype=np.float64)
    for k, c in enumerate(cats[:-1]):
        out[:, k] = (labels == c).astype(np.float64)
    return out


def compute_acss(
    data: SufficiencyData,
    *,
    n_folds: int = 5,
    fdr_alpha: float = 0.05,
    seed: int = 0,
    null: str = "permutation",
    n_perm: int = 1000,
    equivalence_margin: float = 0.1,
) -> ACSSResult:
    """Run the ACSS sufficiency test per metric.

    For each metric :math:`m_j`, fit out-of-fold regressions
    :math:`\\hat{\\mathbb{E}}[L | m_j, A]` and
    :math:`\\hat{\\mathbb{E}}[\\theta | m_j, A]` and examine the residuals
    :math:`\\hat e = L - \\hat{\\mathbb{E}}[L|m_j,A]`,
    :math:`\\hat f = \\theta - \\hat{\\mathbb{E}}[\\theta|m_j,A]`. Under
    :math:`H_0^{(j)}: L \\perp \\theta \\mid (m_j, A)` — the metric *screens off*
    severity from clinical judgement — the residuals are uncorrelated.

    Three quantities come out, and they answer different questions (issue #257):

    * **residual_correlation** ``corr(ê, f̂)`` ∈ [-1, 1] — a signed, bounded
      **effect size**. This is what the ranking is built on.
    * **p_value / p_value_bh** — evidence *against* sufficiency, FDR-controlled
      across the metric pool. Drives the ``defensible`` **decision**, never the
      ordering. Ranking on the BH-adjusted p was a bug: BH is a *monotone
      non-decreasing* transform, so it inverts no pair and adds exactly zero
      ordering information — all it does is **manufacture ties** (8 distinct p
      collapsing to 4 distinct p_bh, largest tie group 4), which the ranker then
      broke by ``MetricSet`` insertion order.
    * **equivalence_p / sufficient_equivalent** — TOST evidence *for* approximate
      sufficiency (``|ρ| < equivalence_margin``). Needed because ``defensible`` is
      a failure-to-reject and therefore partly a statement about N.

    Args:
        data: Likert / severity / metric / anatomy bundle.
        n_folds: K-fold cross-fitting (default 5).
        fdr_alpha: BH-adjusted significance threshold (default 0.05).
        seed: RNG seed.
        null: ``"permutation"`` (**default**) — exact within-anatomy-cell
            permutation null, which sidesteps the ``σ→0`` Wald degeneracy at
            exact sufficiency; or ``"asymptotic"`` (Shah-Peters Gaussian Wald).
            The permutation null was implemented but the shipped ``default_kwargs``
            pinned ``"asymptotic"``, so the degenerate null is what every run used
            (pitfall #15).
        n_perm: number of permutations when ``null="permutation"``.
        equivalence_margin: ``|partial corr|`` below which a metric counts as
            practically sufficient, for the TOST.
    """
    if null not in ("asymptotic", "permutation"):
        raise ValueError(f"unknown null {null!r}; expected 'asymptotic' or 'permutation'")
    if data.anatomy is not None:
        A_onehot = _one_hot(data.anatomy)
    else:
        A_onehot = np.zeros((data.severity.shape[0], 0), dtype=np.float64)

    M = data.metric_values.shape[1]
    N = int(data.severity.shape[0])
    T = np.zeros(M, dtype=np.float64)
    p = np.zeros(M, dtype=np.float64)
    r = np.zeros(M, dtype=np.float64)
    eq_p = np.ones(M, dtype=np.float64)

    for j in range(M):
        m_j = data.metric_values[:, j : j + 1]
        X = np.hstack([m_j, A_onehot]) if A_onehot.size else m_j
        L_hat = _cross_fit_regression(
            X,
            data.likert.astype(np.float64),
            n_folds=n_folds,
            seed=seed,
        )
        theta_hat = _cross_fit_regression(
            X,
            data.severity.astype(np.float64),
            n_folds=n_folds,
            seed=seed,
        )
        e = data.likert - L_hat
        f = data.severity - theta_hat
        if null == "asymptotic":
            T[j], p[j] = _gcm_statistic_and_pvalue(e, f)
        else:
            T[j], p[j] = _gcm_permutation_pvalue(
                e,
                f,
                data.anatomy,
                n_perm=n_perm,
                seed=seed,
            )
        r[j] = _residual_partial_correlation(e, f)
        eq_p[j] = _tost_equivalence_pvalue(r[j], N, equivalence_margin)

    p_bh = _benjamini_hochberg(p)
    defensible = p_bh > fdr_alpha
    sufficient_equivalent = _benjamini_hochberg(eq_p) < fdr_alpha

    return ACSSResult(
        residual_correlation=r,
        gcm_statistic=T,
        p_value=p,
        p_value_bh=p_bh,
        defensible=defensible,
        equivalence_p=eq_p,
        sufficient_equivalent=sufficient_equivalent,
        equivalence_margin=float(equivalence_margin),
        null=null,
        n_samples=N,
    )


# ══════════════════════════════════════════════════════════════════════
# HiBSL — Severity-anchored empirical-Bayes Likert calibration
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class LikertObservations:
    """Radiologist Likert observations + accompanying metric values.

    Attributes:
        likert: ``(N_obs,)`` integer Likert responses ∈ {1, …, K}.
        rater_id: ``(N_obs,)`` rater index ∈ {0, …, R-1}.
        image_id: ``(N_obs,)`` image index ∈ {0, …, N_img-1}.
        severity: ``(N_img,)`` physics-anchored severity θ ∈ [0, 1].
        metric_values: ``(N_img, M)`` per-image metric values, row-
            aligned with ``severity``.
        metric_names: ``(M,)`` names of the metric COLUMNS, in column order.
            Optional for backwards compatibility, but supplying it is what lets
            :class:`HiBSLRanker` bind scores to columns **by name** instead of by
            position. Without it, an external ``--likert-json`` carrying 5 columns
            and a ``MetricSet`` carrying 3 names silently dropped columns 3-4 and
            scored ``ssim`` on the PSNR column (issue #254c).
        anatomy: ``(N_img,)`` integer anatomy/contrast cell ∈ {0, …, A-1},
            or None for a single anatomy cohort.
    """

    likert: np.ndarray
    rater_id: np.ndarray
    image_id: np.ndarray
    severity: np.ndarray
    metric_values: np.ndarray
    metric_names: tuple[str, ...] | None = None
    anatomy: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_obs = self.likert.shape[0]
        if self.rater_id.shape != (n_obs,):
            raise ValueError("rater_id must be shape (N_obs,)")
        if self.image_id.shape != (n_obs,):
            raise ValueError("image_id must be shape (N_obs,)")
        n_img, m = self.metric_values.shape
        if self.severity.shape != (n_img,):
            raise ValueError("severity must be shape (N_img,)")
        if self.anatomy is not None and self.anatomy.shape != (n_img,):
            raise ValueError("anatomy must be shape (N_img,) or None")
        if self.metric_names is not None and len(self.metric_names) != m:
            raise ValueError(
                f"metric_names has {len(self.metric_names)} entries but "
                f"metric_values has {m} columns — the column↔name binding is "
                "ambiguous, so every per-metric score would be attributed to the "
                "wrong metric."
            )


@dataclass(frozen=True)
class HiBSLResult:
    """Per-metric HiBSL calibration result."""

    # (M,) CROSS-FITTED (out-of-fold) mean squared Likert prediction error.
    prediction_mse: np.ndarray
    # (R, K-1) rater-specific cut-points γ_{r, k}. NaN for an excluded rater.
    cutpoints: np.ndarray
    # (R,) rater-specific discriminability β_r. NaN for an excluded rater.
    beta_per_rater: np.ndarray
    # (M,) per-metric monotone-link slope (≈ ∂q/∂m_j at typical m)
    slope_per_metric: np.ndarray
    # Optional posterior samples (always None — there is no MCMC backend).
    posterior_samples: dict[str, np.ndarray] | None
    n_likert_categories: int
    # Log-likelihood, summed over EXACTLY the raters in `fitted_raters` and the
    # `n_observations` observations they contributed. Declared, so an AIC/BIC/LRT
    # built on it knows its own N (issue #254d).
    log_likelihood: float
    # Severity-anchor coefficients (a, b) of the latent-quality prior mean
    # q ~ a + b*theta, and the empirical-Bayes shrinkage weight applied to the
    # Likert evidence. This is where severity enters the model (issue #254a).
    severity_anchor: tuple[float, float] = (0.0, 0.0)
    shrinkage_weight: float = 1.0
    # Raters whose probit was actually fitted / dropped for having < K responses.
    fitted_raters: tuple[int, ...] = ()
    excluded_raters: tuple[int, ...] = ()
    n_observations: int = 0
    n_images: int = 0
    # Laplace-approximation posterior SEs at the MLE point. Shape
    # matches the corresponding mean tensor; NaN for raters with too
    # few observations to invert the Hessian. None when Laplace is
    # disabled.
    cutpoints_se: np.ndarray | None = None
    beta_per_rater_se: np.ndarray | None = None


def _ordered_probit_loglik(
    cutpoints: np.ndarray,
    beta: float,
    q: np.ndarray,
    likert: np.ndarray,
) -> float:
    """Ordered-probit log-likelihood Σ_i log Pr(L_i = k_i | q_i).

    Args:
        cutpoints: ``(K-1,)`` ordered cut-points γ_1 < … < γ_{K-1}.
        beta: Rater discriminability ≥ 0.
        q: ``(N_obs,)`` latent quality per observation.
        likert: ``(N_obs,)`` integer Likert ∈ {1, …, K}.

    Returns:
        Scalar log-likelihood. Uses scipy.stats.norm for the CDF.
    """
    from scipy.stats import norm

    K = cutpoints.shape[0] + 1
    # Pad cutpoints with -∞ / +∞ sentinels so the indexing for k=1
    # (lower bound) and k=K (upper bound) works uniformly.
    pad_low = np.full(1, -np.inf)
    pad_high = np.full(1, np.inf)
    gamma = np.concatenate([pad_low, cutpoints, pad_high])  # (K+1,)

    k_idx = likert.astype(np.int64) - 1  # 0-indexed category
    upper = gamma[k_idx + 1] - beta * q
    lower = gamma[k_idx] - beta * q
    # Pr(L = k) = Φ(upper) - Φ(lower)
    with np.errstate(invalid="ignore"):
        p = norm.cdf(upper) - norm.cdf(lower)
    p = np.clip(p, 1e-12, 1.0)
    return float(np.sum(np.log(p)))


# #329: bound the HiBSL ordered-probit fit to an identifiable box so β and the cutpoints
# cannot run away on quasi-separable rater data (the flat-likelihood / singular-Hessian /
# NaN-SE failure). The box is deliberately tight — a looser bound would let a separable
# rater's β sit AT the boundary where the likelihood is still flat, so verified to keep β
# interior (β ≈ 6-9) across seeds.
_CUTPOINT_BOUND = 10.0
_BETA_BOUND = 10.0


def _fit_per_rater_ordered_probit(
    likert: np.ndarray,
    rater_id: np.ndarray,
    q_obs: np.ndarray,
    K: int,
) -> tuple[np.ndarray, np.ndarray, float, list[int], list[int], int]:
    """Fit per-rater ordered probit by maximum likelihood.

    A rater with fewer than ``K`` responses cannot identify ``K-1`` cut-points.
    Such a rater is **excluded**, not silently handed the hard-coded initialiser
    ``linspace(-1.5, 1.5, K-1)`` and reported as if it had been fitted (issue
    #254d): their cut-points come back ``NaN`` so no downstream code can quietly
    consume a fabricated calibration, and they are named in ``excluded``.

    ``loglik`` is the sum over **exactly** the fitted raters and is returned with
    the observation count it was computed on. Previously the excluded raters were
    skipped in the sum while still appearing in the result, so ``log_likelihood``
    was a sum over an undeclared subset — invalid for AIC/BIC/LRT, which all need
    to know N.

    Returns:
        cutpoints: ``(R, K-1)`` rater-specific γ (NaN rows for excluded raters).
        beta:      ``(R,)`` rater-specific β (NaN for excluded; β_0 = 1 pinned).
        loglik:    Log-likelihood summed over the fitted raters only.
        fitted:    Rater ids that were fitted.
        excluded:  Rater ids dropped for having < K responses.
        n_obs_fit: Number of observations the log-likelihood was computed on.
    """
    from scipy.optimize import minimize

    raters = np.unique(rater_id)
    R = len(raters)
    cutpoints = np.full((R, K - 1), np.nan, dtype=np.float64)
    beta = np.full(R, np.nan, dtype=np.float64)
    total_loglik = 0.0
    fitted: list[int] = []
    excluded: list[int] = []
    n_obs_fit = 0

    # Initial cut-points: equispaced in [-1.5, 1.5] (covers most of N(0,1))
    init_cuts = np.linspace(-1.5, 1.5, K - 1)

    for r_idx, r in enumerate(raters):
        sel = rater_id == r
        q_r = q_obs[sel]
        L_r = likert[sel]
        if q_r.shape[0] < K:
            logger.warning(
                "HiBSL: rater %s has %d responses but K=%d cut-points must be "
                "identified — EXCLUDING them from the fit, the log-likelihood and "
                "the prediction error. Their cut-points are reported as NaN.",
                r,
                q_r.shape[0],
                K,
            )
            excluded.append(int(r))
            continue

        # Identifiability: pin β_0 = 1; only the first rater has β
        # fixed. Other raters get β as a free parameter (per Albert-Chib).
        if r_idx == 0:
            x0 = init_cuts.copy()

            def _neg_ll(x: np.ndarray, q_=q_r, L_=L_r) -> float:
                cuts = np.sort(x)  # enforce ordering
                return -_ordered_probit_loglik(cuts, 1.0, q_, L_)

        else:
            x0 = np.concatenate([init_cuts, [1.0]])

            def _neg_ll(x: np.ndarray, q_=q_r, L_=L_r) -> float:
                cuts = np.sort(x[:-1])
                b = max(float(x[-1]), 1e-6)
                return -_ordered_probit_loglik(cuts, b, q_, L_)

        # Identifiable box (see _CUTPOINT_BOUND / _BETA_BOUND). The reference rater has no
        # free β; a non-reference rater's last parameter is β, floored at 1e-6.
        if r_idx == 0:
            _bounds = [(-_CUTPOINT_BOUND, _CUTPOINT_BOUND)] * (K - 1)
        else:
            _bounds = [(-_CUTPOINT_BOUND, _CUTPOINT_BOUND)] * (K - 1) + [(1e-6, _BETA_BOUND)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                _neg_ll,
                x0,
                method="L-BFGS-B",
                bounds=_bounds,
                options={"maxiter": 2000},
            )
        if r_idx == 0:
            cutpoints[r_idx] = np.sort(res.x)
            beta[r_idx] = 1.0
        else:
            cutpoints[r_idx] = np.sort(res.x[:-1])
            beta[r_idx] = max(float(res.x[-1]), 1e-6)
        total_loglik += -float(res.fun)
        fitted.append(int(r))
        n_obs_fit += int(q_r.shape[0])

    if not fitted:
        raise ValueError(
            f"No rater has the >= K={K} responses needed to identify K-1 "
            "cut-points; HiBSL cannot be calibrated on this cohort."
        )

    return cutpoints, beta, total_loglik, fitted, excluded, n_obs_fit


def _laplace_se_per_rater(
    cutpoints: np.ndarray,
    beta: float,
    q: np.ndarray,
    likert: np.ndarray,
    *,
    fit_beta: bool,
) -> np.ndarray:
    """Laplace-approximation standard errors at the MLE point.

    Computes a numerical Hessian of the negative log-likelihood at
    (cutpoints, β) and returns the square root of the diagonal of
    its inverse — the classical asymptotic SEs from maximum-
    likelihood theory.

    Args:
        cutpoints: ``(K-1,)`` MLE-fit cutpoints.
        beta: MLE-fit β (or 1.0 for the reference rater).
        q: ``(N,)`` latent quality.
        likert: ``(N,)`` integer Likert ∈ {1, …, K}.
        fit_beta: True if β was free in the MLE (non-reference rater);
            False if β was pinned to 1 by the Albert-Chib convention.

    Returns:
        SE vector of length ``K-1`` (cutpoints only) or ``K`` (with β)
        depending on ``fit_beta``. The final entry is the β SE when
        ``fit_beta=True``.
    """
    K_minus_1 = cutpoints.shape[0]
    params = np.concatenate([cutpoints, [beta]]) if fit_beta else cutpoints
    n_p = len(params)

    if q.shape[0] < n_p + 2:
        return np.full(n_p, np.nan)

    def _neg_ll(p: np.ndarray) -> float:
        cuts = np.sort(p[:K_minus_1])
        b = max(float(p[K_minus_1]), 1e-6) if fit_beta else 1.0
        return -_ordered_probit_loglik(cuts, b, q, likert)

    # Central-difference Hessian (cheap; we only need diag for SEs)
    eps = 1e-3
    hess = np.zeros((n_p, n_p), dtype=np.float64)
    f0 = _neg_ll(params)
    for i in range(n_p):
        for j in range(i, n_p):
            pp = params.copy()
            pp[i] += eps
            pp[j] += eps
            fpp = _neg_ll(pp)
            pm = params.copy()
            pm[i] += eps
            pm[j] -= eps
            fpm = _neg_ll(pm)
            mp = params.copy()
            mp[i] -= eps
            mp[j] += eps
            fmp = _neg_ll(mp)
            mm = params.copy()
            mm[i] -= eps
            mm[j] -= eps
            fmm = _neg_ll(mm)
            hij = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
            hess[i, j] = hij
            hess[j, i] = hij

    # Pseudo-inverse to tolerate near-singular Hessians
    try:
        cov = np.linalg.pinv(hess)
        diag = np.diag(cov)
        return np.sqrt(np.where(diag > 0, diag, np.nan))
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.full(n_p, np.nan)


def _predict_likert_expected(
    cutpoints: np.ndarray,
    beta: float,
    q: np.ndarray,
) -> np.ndarray:
    """Per-observation expected Likert E[L | q] for a single rater.

    Returns a real-valued posterior mean rather than the modal
    category — closer to the L1 / MSE objective and smoother under
    cross-validation.
    """
    from scipy.stats import norm

    K = cutpoints.shape[0] + 1
    pad_low = np.full(1, -np.inf)
    pad_high = np.full(1, np.inf)
    gamma = np.concatenate([pad_low, cutpoints, pad_high])

    # (K, N): Pr(L = k | q)
    upper = gamma[1:].reshape(-1, 1) - beta * q.reshape(1, -1)
    lower = gamma[:-1].reshape(-1, 1) - beta * q.reshape(1, -1)
    with np.errstate(invalid="ignore"):
        p = norm.cdf(upper) - norm.cdf(lower)
    p = np.clip(p, 0.0, 1.0)
    p /= p.sum(axis=0, keepdims=True).clip(min=1e-12)

    categories = np.arange(1, K + 1, dtype=np.float64).reshape(-1, 1)
    return (categories * p).sum(axis=0)


def _isotonic_quality_from_metric(
    metric_values_per_image: np.ndarray,
    quality_per_image: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit isotonic g_j: m_j → q via PAV **in-sample**, return q-hat and slope.

    In-sample: the returned ``q_hat`` is the fit evaluated on its own training
    points. Use :func:`_crossfit_isotonic_quality` for anything that gets scored —
    an in-sample isotonic fit hands a free advantage to whichever metric can
    overfit hardest, i.e. the noisiest one (issue #254f). This function remains for
    the ``slope_per_metric`` summary, which is descriptive, not a score.

    A metric that crashed on its cohort (non-finite values) yields **NaN**, never a
    zero-filled stand-in: ``q_hat = zeros`` gave a crashed metric a finite,
    respectable-looking MSE instead of propagating the crash (issue #254e).

    The slope is the mean numerical derivative of the fitted function — a scalar
    summary of the metric's effective discriminability.
    """
    from sklearn.isotonic import IsotonicRegression

    n = quality_per_image.shape[0]
    finite = np.isfinite(metric_values_per_image) & np.isfinite(quality_per_image)
    if not np.any(finite):
        return np.full(n, np.nan), float("nan")

    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    q_hat = np.full(n, np.nan, dtype=np.float64)
    try:
        iso.fit(metric_values_per_image[finite], quality_per_image[finite])
        q_hat[finite] = iso.predict(metric_values_per_image[finite])
    except ValueError as exc:
        # A genuinely un-fittable column (degenerate input). Propagate as NaN so
        # the metric ranks LAST, rather than laundering it into a finite score.
        logger.warning("HiBSL: isotonic link failed (%s) — metric scored NaN.", exc)
        return np.full(n, np.nan), float("nan")

    # Slope summary: mean |dq/dm| over the fitted sample.
    m_f = metric_values_per_image[finite]
    sorted_idx = np.argsort(m_f)
    m_s = m_f[sorted_idx]
    q_s = q_hat[finite][sorted_idx]
    dm = np.diff(m_s)
    dq = np.diff(q_s)
    valid = np.abs(dm) > 1e-12
    slope = float(np.mean(np.abs(dq[valid] / dm[valid]))) if np.any(valid) else 0.0

    return q_hat, slope


def _crossfit_isotonic_quality(
    metric_values_per_image: np.ndarray,
    quality_per_image: np.ndarray,
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold isotonic link predictions ``ĝ_j(m_j)`` — the honest q̂.

    The isotonic link is **metric-specific**, so any in-sample optimism it carries
    is proportional to how well that particular metric can overfit the latent
    quality — and a pure-noise metric overfits best. Fitting and scoring on the
    same points therefore ranked a pure-noise metric ABOVE a constant metric even
    though both carry exactly zero information (measured: MSE 1.532 vs 1.585,
    issue #254f). K-fold cross-fitting removes that advantage: every image's ``q̂``
    comes from a link fitted without it.

    (The per-rater probit is deliberately *not* cross-fitted. It is metric-
    independent — the same cut-points serve every metric — so whatever in-sample
    optimism it carries is common to the whole pool and cannot reorder it.)

    Returns:
        ``(N_img,)`` out-of-fold ``q̂``; ``NaN`` wherever the metric is non-finite.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import KFold

    n = quality_per_image.shape[0]
    q_hat = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(metric_values_per_image) & np.isfinite(quality_per_image)
    idx_finite = np.flatnonzero(finite)
    if idx_finite.size < 4:
        return q_hat

    k = int(min(n_folds, idx_finite.size))
    if k < 2:
        return q_hat
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for train_pos, test_pos in kf.split(idx_finite):
        tr = idx_finite[train_pos]
        te = idx_finite[test_pos]
        iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        try:
            iso.fit(metric_values_per_image[tr], quality_per_image[tr])
            q_hat[te] = iso.predict(metric_values_per_image[te])
        except ValueError:
            # Degenerate fold — leave the fold's q̂ NaN; it propagates to the MSE.
            continue
    return q_hat


def fit_hibsl(
    obs: LikertObservations,
    *,
    n_likert_categories: int | None = None,
    use_numpyro: bool = False,
    compute_laplace_se: bool = True,
    n_folds: int = 5,
    seed: int = 0,
) -> HiBSLResult:
    r"""Fit the severity-anchored empirical-Bayes Likert calibration.

    The model (and where severity enters — issue #254a)
    ---------------------------------------------------
    .. math::

        q_i \;\sim\; \mathcal N\!\bigl(a + b\,\theta_i,\ \sigma_u^2\bigr)
        \qquad\text{(severity-anchored latent-quality prior)}

        L_{ri} \mid q_i \;\sim\; \mathrm{OrderedProbit}(\gamma_r,\ \beta_r,\ q_i)

    The per-image latent quality is a **partially-pooled** (empirical-Bayes)
    combination of what the raters said about image *i* and what the physics
    severity θ_i predicts they should have said:

    .. math::

        \hat q_i = \mu_i + w_i\,(z_i - \mu_i), \qquad
        \mu_i = a + b\,\theta_i, \qquad
        w_i = \frac{\sigma_u^2}{\sigma_u^2 + \sigma_e^2 / n_i}

    with ``z_i`` the standardised mean Likert for image *i*, ``n_i`` its number of
    ratings, ``σ_e²`` the between-rater variance and ``σ_u²`` the image-level
    variance around the severity line. Sparsely-rated images shrink toward the
    severity prediction; heavily-rated ones trust their raters. An image with **no**
    ratings gets ``μ_i`` — the severity prior — rather than the old ``0.0``.

    Before this, ``obs.severity`` was touched exactly once, for
    ``N_img = obs.severity.shape[0]``. Replacing it with zeros or with noise left
    ``prediction_mse``, ``log_likelihood`` and ``cutpoints`` **bitwise identical** —
    a "severity-to-Likert calibration" in which severity was not in the model at
    all (pitfall #16).

    What is and is not Bayesian (issue #254b)
    -----------------------------------------
    This is **empirical Bayes**: the latent quality is partially pooled toward a
    fitted prior, but the per-rater probits are fit by maximum likelihood,
    independently, with **no pooling across raters** and no posterior. There is no
    MCMC backend: ``use_numpyro=True`` raises :class:`NotImplementedError` rather
    than returning normally with ``posterior_samples=None`` while logging the false
    diagnosis "numpyro not available" (the stub never imported numpyro; it raised
    ``ImportError`` unconditionally and the caller swallowed it). Consequently
    there are no R-hat / ESS diagnostics to report, and none are claimed.

    Args:
        obs: Bundled Likert + metric data.
        n_likert_categories: Number of Likert categories K. Inferred
            from ``obs.likert`` if None.
        use_numpyro: Must be False. True raises — there is no MCMC backend.
        compute_laplace_se: Compute Laplace SEs at the MLE point.
        n_folds: K-fold cross-fitting of the per-metric isotonic link (#254f).
        seed: RNG seed for the cross-fitting folds.

    Raises:
        NotImplementedError: If ``use_numpyro=True``.
        ValueError: If K < 2, if no image carries a Likert observation, or if no
            rater has enough responses to identify the cut-points.
    """
    if use_numpyro:
        raise NotImplementedError(
            "HiBSL has no MCMC/NUTS backend: there is no numpyro model, no "
            "posterior, and therefore no R-hat/ESS. use_numpyro=True previously "
            "returned normally with posterior_samples=None while logging 'numpyro "
            "not available' — a false diagnosis blaming the environment for a "
            "hard-coded stub (issue #254b, pitfall #9: no silent fallbacks). Use "
            "use_numpyro=False for the empirical-Bayes / MLE backend that exists."
        )

    K = int(obs.likert.max()) if n_likert_categories is None else int(n_likert_categories)
    if K < 2:
        raise ValueError("Need at least K=2 Likert categories")

    # ── Step 1: per-image Likert evidence ──
    N_obs = obs.likert.shape[0]
    N_img = obs.severity.shape[0]
    likert_mean_per_img = np.full(N_img, np.nan, dtype=np.float64)
    counts = np.zeros(N_img, dtype=np.int64)
    sums = np.zeros(N_img, dtype=np.float64)
    sumsq = np.zeros(N_img, dtype=np.float64)
    for n in range(N_obs):
        i = int(obs.image_id[n])
        sums[i] += obs.likert[n]
        sumsq[i] += float(obs.likert[n]) ** 2
        counts[i] += 1
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("No images have any Likert observations")
    likert_mean_per_img[valid] = sums[valid] / counts[valid]

    # Standardise onto a z-scale so the cut-points sit in a stable range.
    mu = float(np.nanmean(likert_mean_per_img[valid]))
    sd = float(np.nanstd(likert_mean_per_img[valid]))
    if sd < 1e-9:
        sd = 1.0
    z = np.zeros(N_img, dtype=np.float64)
    z[valid] = (likert_mean_per_img[valid] - mu) / sd

    # ── Step 2: the severity anchor — regress the Likert evidence on θ ──
    theta = np.asarray(obs.severity, dtype=np.float64)
    obs_idx = np.flatnonzero(valid & np.isfinite(theta))
    if obs_idx.size >= 2 and float(np.std(theta[obs_idx])) > 1e-12:
        b_sev, a_sev = np.polyfit(theta[obs_idx], z[obs_idx], 1)
    else:
        # No usable severity contrast: the prior mean degenerates to the grand
        # mean and the EB step becomes a no-op (w = 1). Reported, not hidden.
        b_sev, a_sev = 0.0, 0.0
        logger.warning(
            "HiBSL: severity has no variation across the rated images — the "
            "severity anchor is degenerate and the latent quality falls back to "
            "the Likert evidence alone."
        )
    prior_mean = a_sev + b_sev * theta  # μ_i

    # Variance components, on the z-scale.
    # σ_e²: between-rater (within-image) variance; only images with n_i >= 2 inform it.
    multi = counts >= 2
    if np.any(multi):
        within_var = (sumsq[multi] - (sums[multi] ** 2) / counts[multi]) / np.maximum(
            counts[multi] - 1, 1
        )
        sigma_e_sq = float(np.mean(within_var)) / (sd**2)
    else:
        sigma_e_sq = 0.0
    # σ_u²: image-level variance AROUND the severity line, net of the measurement
    # noise already inside z (method of moments; floored at 0).
    resid = z[obs_idx] - prior_mean[obs_idx]
    meas_var = float(np.mean(sigma_e_sq / np.maximum(counts[obs_idx], 1)))
    sigma_u_sq = max(float(np.var(resid)) - meas_var, 0.0)

    # ── Step 3: empirical-Bayes latent quality (this is where θ acts) ──
    q_per_img = prior_mean.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.zeros(N_img, dtype=np.float64)
        denom = sigma_u_sq + sigma_e_sq / np.maximum(counts, 1)
        nz = valid & (denom > 1e-12)
        w[nz] = sigma_u_sq / denom[nz]
        # Degenerate variance components (e.g. a single rater, so σ_e² is
        # unestimable) ⇒ trust the raters fully rather than invent shrinkage.
        w[valid & ~nz] = 1.0
    q_per_img[valid] = prior_mean[valid] + w[valid] * (z[valid] - prior_mean[valid])
    # Unrated images keep the severity prior μ_i — not a zero-impute.
    shrinkage_weight = float(np.mean(w[valid])) if np.any(valid) else 1.0

    q_obs = q_per_img[obs.image_id]

    # ── Step 4: per-rater ordered probit by MLE ──
    (
        cutpoints,
        beta,
        loglik,
        fitted_raters,
        excluded_raters,
        n_obs_fit,
    ) = _fit_per_rater_ordered_probit(obs.likert, obs.rater_id, q_obs, K)
    fitted_set = set(fitted_raters)

    # Optional: Laplace-approximation SEs at the MLE point.
    cutpoints_se: np.ndarray | None = None
    beta_per_rater_se: np.ndarray | None = None
    if compute_laplace_se:
        R = beta.shape[0]
        cutpoints_se = np.full_like(cutpoints, np.nan)
        beta_per_rater_se = np.full(R, np.nan)
        for r_idx, r in enumerate(np.unique(obs.rater_id)):
            if int(r) not in fitted_set:
                continue
            sel = obs.rater_id == r
            q_r = q_obs[sel]
            L_r = obs.likert[sel]
            if q_r.shape[0] < K + 2:
                continue
            fit_beta = r_idx != 0  # reference rater has β pinned to 1
            se_vec = _laplace_se_per_rater(
                cutpoints[r_idx],
                beta[r_idx],
                q_r,
                L_r,
                fit_beta=fit_beta,
            )
            if fit_beta:
                cutpoints_se[r_idx] = se_vec[: K - 1]
                beta_per_rater_se[r_idx] = se_vec[K - 1]
            else:
                cutpoints_se[r_idx] = se_vec[: K - 1]
                # β is pinned, not a free parameter — leave SE as NaN.

    # ── Step 5: per-metric isotonic link g_j: m_j → q, CROSS-FITTED ──
    M = obs.metric_values.shape[1]
    slopes = np.full(M, np.nan, dtype=np.float64)
    mse = np.full(M, np.nan, dtype=np.float64)

    for j in range(M):
        _, slopes[j] = _isotonic_quality_from_metric(obs.metric_values[:, j], q_per_img)
        q_hat = _crossfit_isotonic_quality(
            obs.metric_values[:, j], q_per_img, n_folds=n_folds, seed=seed
        )

        # Held-out predicted Likert under q̂, over the FITTED raters only.
        sq_err_total = 0.0
        n_total = 0
        for r_idx, r in enumerate(np.unique(obs.rater_id)):
            if int(r) not in fitted_set:
                continue
            sel = obs.rater_id == r
            q_obs_r = q_hat[obs.image_id[sel]]
            L_true = obs.likert[sel].astype(np.float64)
            keep = np.isfinite(q_obs_r)
            if not np.any(keep):
                continue
            L_pred = _predict_likert_expected(cutpoints[r_idx], beta[r_idx], q_obs_r[keep])
            sq_err_total += float(np.sum((L_true[keep] - L_pred) ** 2))
            n_total += int(keep.sum())
        # A metric with no usable out-of-fold prediction anywhere (crashed column)
        # keeps its NaN — it must never be laundered into a finite score.
        mse[j] = sq_err_total / n_total if n_total else float("nan")

    return HiBSLResult(
        prediction_mse=mse,
        cutpoints=cutpoints,
        beta_per_rater=beta,
        slope_per_metric=slopes,
        # No MCMC backend exists, so there is never a posterior. `use_numpyro=True`
        # raised at the top of this function rather than reaching here.
        posterior_samples=None,
        n_likert_categories=K,
        log_likelihood=loglik,
        severity_anchor=(float(a_sev), float(b_sev)),
        shrinkage_weight=shrinkage_weight,
        fitted_raters=tuple(fitted_raters),
        excluded_raters=tuple(excluded_raters),
        n_observations=int(n_obs_fit),
        n_images=int(N_img),
        cutpoints_se=cutpoints_se,
        beta_per_rater_se=beta_per_rater_se,
    )


def _numpyro_posterior(
    obs: LikertObservations,
    K: int,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Not implemented. There is no MCMC backend — see :func:`fit_hibsl`.

    Kept only so the old import site fails with an honest message. The previous
    body raised ``ImportError("numpyro path not implemented in this minimal
    build")`` *without ever importing numpyro*, and ``fit_hibsl`` caught it and
    logged "numpyro not available" — pinning a hard-coded stub on the user's
    environment (issue #254b).
    """
    raise NotImplementedError(
        "HiBSL has no numpyro/NUTS backend — no model, no posterior, no R-hat/ESS. "
        "This is not an environment problem: installing numpyro would change "
        "nothing. Use the empirical-Bayes / MLE backend (use_numpyro=False)."
    )


# ══════════════════════════════════════════════════════════════════════
# BaseRanker wrappers — external data injected at __init__, no fallback.
# ══════════════════════════════════════════════════════════════════════


def _severity_vector(dataset: MetricEvaluationDataset) -> np.ndarray:
    """Per-sample physics-anchored severity θ, aligned with ``dataset.samples``."""
    return np.array(
        [float(s.severity.get("theta", 0.0)) for s in dataset.samples],
        dtype=np.float64,
    )


def _metric_matrix(
    metric_set: MetricSet, dataset: MetricEvaluationDataset, names: Sequence[str]
) -> np.ndarray:
    """``(N, M)`` matrix whose column ``j`` is ``dataset.metric_values[names[j]]``."""
    return np.column_stack([np.asarray(dataset.metric_values[n], dtype=np.float64) for n in names])


@register_ranker(
    "dcms",
    generation=4,
    requires_task_net=True,
    data_kwargs=("task_performance",),
    description=(
        "Causal mediation decomposition (parametric VanderWeele closed form). "
        "Splits the effect of severity on frozen-task performance into the part "
        "routed through the metric (NIE), the direct part (NDE) and the mediated "
        "interaction; ranks on the BOUNDED mediation fraction."
    ),
    default_kwargs={"n_pairs": 256, "seed": 0},
)
class DCMSRanker(BaseRanker):
    """Gen-4 mediation ranker (capability-gated: needs measured ``task_performance``).

    ``task_performance`` must be the output of a **real** frozen downstream task
    network evaluated on the degraded images. It must never be a curve synthesised
    from the severity grid: if ``P = f(θ)`` exactly, then ``P ⟂ m | θ`` for EVERY
    metric, the true natural indirect effect through every metric is identically
    zero, and any non-zero mediation this estimator reports is pure outcome-model
    misspecification. Fed such a curve it ranked a **constant, zero-information
    metric first** (issue #253).
    """

    method_name = "DCMS"

    def __init__(
        self,
        *,
        task_performance: np.ndarray | Sequence[float] | None = None,
        anatomy: np.ndarray | Sequence[int] | None = None,
        n_pairs: int = 256,
        seed: int = 0,
    ) -> None:
        self.task_performance = task_performance
        self.anatomy = anatomy
        self.n_pairs = n_pairs
        self.seed = seed

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        names = metric_set.names()
        if self.task_performance is None:
            raise ValueError(
                "DCMSRanker requires task_performance measured by a frozen "
                "downstream task network on the degraded images. It has no default "
                "and a severity-derived synthetic curve is NOT a substitute (with "
                "P = f(theta) the true indirect effect through every metric is "
                "identically zero — issue #253). Inject it via "
                "MetaEvaluationPipeline.from_registry(ranker_kwargs={'dcms': "
                "{'task_performance': ...}}), or leave have_task_net=False so this "
                "ranker is omitted from the pool."
            )
        task_perf = np.asarray(self.task_performance, dtype=float)
        if task_perf.shape[0] != dataset.n_samples:
            raise ValueError(
                f"task_performance has {task_perf.shape[0]} entries but the dataset "
                f"has {dataset.n_samples} samples — they must be row-aligned."
            )
        severity = _severity_vector(dataset)
        metric_values = _metric_matrix(metric_set, dataset, names)
        anatomy = None if self.anatomy is None else np.asarray(self.anatomy)
        res = compute_dcms(
            MediationData(
                severity=severity,
                metric_values=metric_values,
                task_performance=task_perf,
                anatomy=anatomy,
            ),
            n_pairs=self.n_pairs,
            seed=self.seed,
        )
        # Bounded mediation fraction, NOT the unbounded proportion-mediated: under
        # suppression PM explodes (+54 on a measured example) and a metric whose
        # direct and indirect effects merely cancel outranks an honestly fully-
        # mediating one (issue #255b). NaN (crashed metric) propagates and ranks
        # LAST — it is not laundered to 0.0 (issue #255c).
        scores = {names[j]: float(res.mediation_fraction[j]) for j in range(len(names))}
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "nie": {names[j]: float(res.nie[j]) for j in range(len(names))},
                "nde": {names[j]: float(res.nde[j]) for j in range(len(names))},
                "interaction": {names[j]: float(res.interaction[j]) for j in range(len(names))},
                "total_effect": {names[j]: float(res.total_effect[j]) for j in range(len(names))},
                "proportion_mediated": {
                    names[j]: float(res.proportion_mediated[j]) for j in range(len(names))
                },
                # Identification is ASSUMED, not tested. rho_zero says how much
                # mediator/outcome error correlation would nullify the NIE.
                "rho_zero": {names[j]: float(res.rho_zero[j]) for j in range(len(names))},
                "identification": "sequential ignorability ASSUMED, not tested",
                "positivity": res.positivity,
                "n_pairs": res.n_pairs,
            },
        )


@register_ranker(
    "acss",
    generation=4,
    requires_likert=True,
    data_kwargs=("likert",),
    description=(
        "Anatomy-Conditional Sufficient Statistic via Shah-Peters GCM. Tests "
        "whether the metric screens off severity from Likert. Ranks on the BOUNDED "
        "residual partial correlation (an effect size); BH-FDR drives the "
        "defensibility DECISION only, plus a TOST equivalence certificate."
    ),
    default_kwargs={
        "n_folds": 5,
        "fdr_alpha": 0.05,
        "seed": 0,
        "null": "permutation",
        "n_perm": 1000,
        "equivalence_margin": 0.1,
    },
)
class ACSSRanker(BaseRanker):
    """Gen-4 sufficiency ranker (capability-gated: needs radiologist ``likert``)."""

    method_name = "ACSS"

    def __init__(
        self,
        *,
        likert: np.ndarray | Sequence[float] | None = None,
        anatomy: np.ndarray | Sequence[int] | None = None,
        n_folds: int = 5,
        fdr_alpha: float = 0.05,
        seed: int = 0,
        null: str = "permutation",
        n_perm: int = 1000,
        equivalence_margin: float = 0.1,
    ) -> None:
        self.likert = likert
        self.anatomy = anatomy
        self.n_folds = n_folds
        self.fdr_alpha = fdr_alpha
        self.seed = seed
        self.null = null
        self.n_perm = n_perm
        self.equivalence_margin = equivalence_margin

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        names = metric_set.names()
        if self.likert is None:
            raise ValueError(
                "ACSSRanker requires radiologist likert scores; inject via "
                "MetaEvaluationPipeline.from_registry(ranker_kwargs={'acss': "
                "{'likert': ...}}), or leave have_likert=False so this ranker is "
                "omitted from the pool."
            )
        likert = np.asarray(self.likert, dtype=float)
        if likert.shape[0] != dataset.n_samples:
            raise ValueError(
                f"likert has {likert.shape[0]} entries but the dataset has "
                f"{dataset.n_samples} samples — they must be row-aligned."
            )
        severity = _severity_vector(dataset)
        metric_values = _metric_matrix(metric_set, dataset, names)
        anatomy = None if self.anatomy is None else np.asarray(self.anatomy)
        res = compute_acss(
            SufficiencyData(
                likert=likert,
                severity=severity,
                metric_values=metric_values,
                anatomy=anatomy,
            ),
            n_folds=self.n_folds,
            fdr_alpha=self.fdr_alpha,
            seed=self.seed,
            null=self.null,
            n_perm=self.n_perm,
            equivalence_margin=self.equivalence_margin,
        )
        # Rank on the SIGNED, BOUNDED effect size, not on the BH-adjusted p-value.
        # BH is monotone non-decreasing: it inverts no pair, so it contributes zero
        # ordering information and only manufactures ties, which then break by
        # MetricSet insertion order (issue #257a). Less residual dependence left
        # after conditioning on the metric = closer to sufficient = higher score.
        scores = {names[j]: float(-abs(res.residual_correlation[j])) for j in range(len(names))}
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "residual_correlation": {
                    names[j]: float(res.residual_correlation[j]) for j in range(len(names))
                },
                "gcm_statistic": {names[j]: float(res.gcm_statistic[j]) for j in range(len(names))},
                "p_value": {names[j]: float(res.p_value[j]) for j in range(len(names))},
                "p_value_bh": {names[j]: float(res.p_value_bh[j]) for j in range(len(names))},
                # DECISION (FDR-controlled), never the ordering. A failure to
                # reject — so partly a statement about N.
                "defensible": {names[j]: bool(res.defensible[j]) for j in range(len(names))},
                # POSITIVE evidence for sufficiency: TOST rejects |rho| >= margin.
                "equivalence_p": {names[j]: float(res.equivalence_p[j]) for j in range(len(names))},
                "sufficient_equivalent": {
                    names[j]: bool(res.sufficient_equivalent[j]) for j in range(len(names))
                },
                "equivalence_margin": res.equivalence_margin,
                "null": res.null,
                "n_samples": res.n_samples,
            },
        )


@register_ranker(
    "hibsl",
    generation=4,
    requires_likert=True,
    data_kwargs=("observations",),
    description=(
        "Severity-anchored empirical-Bayes Likert calibration. Image-level latent "
        "quality is partially pooled toward a severity-regression prior; per-rater "
        "ordered probits are fit by ML (no pooling across raters, no MCMC). Reports "
        "the CROSS-FITTED per-metric Likert prediction error."
    ),
    default_kwargs={"use_numpyro": False, "n_folds": 5, "seed": 0},
)
class HiBSLRanker(BaseRanker):
    """Gen-4 calibration ranker (capability-gated: needs Likert ``observations``)."""

    method_name = "HiBSL"

    def __init__(
        self,
        *,
        observations: LikertObservations | None = None,
        use_numpyro: bool = False,
        n_folds: int = 5,
        seed: int = 0,
    ) -> None:
        self.observations = observations
        self.use_numpyro = use_numpyro
        self.n_folds = n_folds
        self.seed = seed

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)
        names = metric_set.names()
        obs = self.observations
        if obs is None:
            raise ValueError(
                "HiBSLRanker requires radiologist Likert observations; inject via "
                "MetaEvaluationPipeline.from_registry(ranker_kwargs={'hibsl': "
                "{'observations': ...}}), or leave have_likert=False so this ranker "
                "is omitted from the pool."
            )

        # Bind score columns to metric names BY NAME (issue #254c). The old code
        # assumed observations.metric_values[:, j] == metric_set.names()[j] with no
        # check at all: an external --likert-json carrying 5 columns against a
        # 3-name MetricSet silently dropped columns 3-4 and scored 'ssim' on the
        # PSNR column. There is no way to recover the intended binding after the
        # fact, so a mismatch must RAISE.
        n_cols = obs.metric_values.shape[1]
        if obs.metric_names is None:
            if n_cols != len(names):
                raise ValueError(
                    f"LikertObservations.metric_values has {n_cols} columns but the "
                    f"MetricSet has {len(names)} metrics ({list(names)}), and the "
                    "observations carry no metric_names to disambiguate. Column j "
                    "would be attributed to names[j] positionally — silently "
                    "dropping the extra columns and scoring the wrong metric. Set "
                    "LikertObservations.metric_names to bind columns by name."
                )
            col_of = dict(zip(names, range(n_cols), strict=True))
        else:
            col_of = {nm: j for j, nm in enumerate(obs.metric_names)}
            missing = [nm for nm in names if nm not in col_of]
            if missing:
                raise ValueError(
                    f"LikertObservations carries no column for metric(s) {missing}; "
                    f"it has {list(obs.metric_names)}. Every metric in the ranking "
                    "pool must have an observed column — scoring it on some other "
                    "metric's column is worse than not scoring it."
                )

        res = fit_hibsl(
            obs,
            use_numpyro=self.use_numpyro,
            n_folds=self.n_folds,
            seed=self.seed,
        )
        # Higher = better; HiBSL is a lower-is-better MSE, so negate. The MSE is
        # CROSS-FITTED (issue #254f), and a crashed metric keeps its NaN and ranks
        # LAST (issue #254e).
        scores = {nm: float(-res.prediction_mse[col_of[nm]]) for nm in names}
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "prediction_mse": {nm: float(res.prediction_mse[col_of[nm]]) for nm in names},
                "slope_per_metric": {nm: float(res.slope_per_metric[col_of[nm]]) for nm in names},
                "n_likert_categories": res.n_likert_categories,
                # Declared N: the log-likelihood is a sum over EXACTLY these
                # raters/observations (issue #254d).
                "log_likelihood": res.log_likelihood,
                "n_observations": res.n_observations,
                "n_images": res.n_images,
                "fitted_raters": list(res.fitted_raters),
                "excluded_raters": list(res.excluded_raters),
                # Provenance of the severity anchor — proof that theta is IN the
                # model (issue #254a).
                "severity_anchor": {
                    "intercept": res.severity_anchor[0],
                    "slope": res.severity_anchor[1],
                },
                "shrinkage_weight": res.shrinkage_weight,
                "backend": "empirical_bayes_mle",
            },
        )


__all__ = [
    # Rankers
    "DCMSRanker",
    "ACSSRanker",
    "HiBSLRanker",
    # Data classes
    "MediationData",
    "SufficiencyData",
    "LikertObservations",
    # compute_* / fit_*
    "compute_dcms",
    "baron_kenny_proportion_mediated",
    "compute_acss",
    "fit_hibsl",
    # Result classes
    "DCMSResult",
    "ACSSResult",
    "HiBSLResult",
]
