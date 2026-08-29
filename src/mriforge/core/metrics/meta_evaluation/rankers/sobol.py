"""Gen-2 Sobol ranker — variance-based response-ratio surrogate.

This is the **production default** Sobol ranker: the full Saltelli A/B/AB sweep
(``scripts/sim2rank/meta_eval.py::compute_sobol_indices``) needs a live,
re-evaluable metric function, and the meta-evaluation dataset only carries
pre-computed metric scalars — so production always takes this fallback. The
Saltelli estimator itself is correct (validated against Ishigami) and is not
touched here; what this module reimplements is the **surrogate + the aggregate**.

Score
-----
For each metric, per degradation **family** (the legacy "axis") :math:`j`, the
total-effect surrogate is the fraction of the trajectory's variance explained by
its severity trend,

.. math::
    S_{T,j} \\;=\\; \\eta^2_j \\;=\\; 1 - \\frac{\\hat\\sigma^2_\\varepsilon}
                                        {\\mathrm{Var}_t\\, \\bar m_j(s_t)},
    \\qquad
    \\hat\\sigma^2_\\varepsilon = \\tfrac12\\,\\overline{(\\Delta_t \\bar m_j)^2},

and the score combines *magnitude* with *specialisation*:

.. math::
    \\mathcal R_k = \\max_j S_{T,j}^k \\cdot
        \\bigl[(1 - w) + w\\,A_k\\bigr] \\in [0, 1],
    \\qquad A_k = 1 - H(q^k)/\\log F ,

with :math:`q = S_T/\\sum S_T` the normalised profile. A ``target_profile`` swaps
the specialisation term for profile *alignment*, ``A = clip(1 − β‖q − t‖₂, 0, 1)``
(critique §22/L15), which is what those two knobs were always for.

.. warning:: **What this replaces (issue #249).**

   *The aggregate was the normalised **entropy** of the total-effect profile* —
   i.e. it rewarded **evenness**. That is exactly backwards: a metric that measures
   *nothing* responds to every factor equally (``S_T = 1`` on all of them, a
   perfectly even profile) and therefore took the **maximum** score. On a
   3-metric probe the ranking came out ``pure_noise (0.9989) > broad_additive
   (0.9988) > signal_1factor (0.0000)`` — noise first, signal last. Breadth is the
   Task and MS³ axes' job; this axis now asks *does the metric respond at all*
   (``max_j S_T``) and *is that response specific to a degradation family*.

   *The per-axis surrogate was* ``|SROCC|``, *which is rank-based* and therefore
   saturates at exactly ``1.0`` for **every** strictly monotone metric: the pool
   ties at the ceiling, the profile is uniform for everyone, and the ranking
   inverted (``pure_noise 0.9941`` above ``saturating_good 0.9675``). The variance
   ratio above does not saturate — white noise gives ``η² = 0``, a clean monotone
   response ``1 − 6/(T² − 1)``, and a noisy-but-monotone metric lands in between,
   which is the discrimination ``|SROCC|`` could not make.

   The interaction gate (``S_T − S``) that the live Saltelli path uses to detect
   the noise signature is **not estimable here** — the severity sweep is not a
   factorial design, one family is active per sample. This module says so
   (``diagnostics["interaction_gap_estimable"] = False``) instead of fabricating a
   zero gap; the noise metric is rejected on magnitude (``η² ≈ 0``) instead.

``coverage`` (the normalised entropy) is retained as a **diagnostic** — it is a
real property of the profile, it was just never a defensible score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import (
    _build_trajectory_table,
    _per_family_average,
)

_EPS = 1e-12


@dataclass
class SobolConfig:
    """Configuration for the Sobol (variance-based response-ratio) ranker.

    Attributes:
        beta: Penalty weight on the L2 deviation from ``target_profile`` (only used
            when a target profile is supplied).
        target_profile: Optional length-``n_families`` desired total-effect profile.
            ``None`` (default) scores specialisation; a supplied profile scores
            alignment to it. Validated on use — a wrong-length / negative /
            all-zero profile RAISES rather than being silently ignored.
        specialisation_weight: ``w`` — how much of the score rides on the profile
            vs. on raw response magnitude. ``0.0`` → magnitude only.
    """

    beta: float = 0.5
    target_profile: list[float] | None = None
    specialisation_weight: float = 0.5


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


def severity_response_ratio(trajectory: torch.Tensor) -> float:
    r"""Variance of a trajectory explained by its severity trend, in ``[0, 1]``.

    ``η² = 1 − ½·mean(Δm²) / Var(m)`` — the lag-1 successive-difference estimate of
    the white-noise floor over the trajectory's total variance. White noise → 0; a
    clean monotone response → ``1 − 6/(T² − 1)`` (0.60 at ``T = 4``, 0.90 at
    ``T = 8``); a noisy monotone response lands in between. Constant or
    shorter-than-3 trajectories → 0.0.

    Mirrors ``scripts/sim2rank/meta_eval.py::severity_response_ratio`` exactly.
    """
    m = np.nan_to_num(trajectory.detach().cpu().numpy().astype(np.float64).ravel())
    if m.size < 3:
        return 0.0
    var = float(m.var())
    if var < 1e-24:
        return 0.0
    sigma2 = 0.5 * float(np.mean(np.diff(m) ** 2))
    return float(np.clip(1.0 - sigma2 / var, 0.0, 1.0))


@register_ranker(
    "sobol",
    generation=2,
    description="Sobol severity-response magnitude x specialisation (variance-based)",
)
class SobolRanker(BaseRanker):
    method_name = "Sobol"

    def __init__(self, config: SobolConfig | None = None) -> None:
        self.config = config or SobolConfig()
        if not 0.0 <= self.config.specialisation_weight <= 1.0:
            raise ValueError(
                "specialisation_weight must lie in [0, 1], got "
                f"{self.config.specialisation_weight!r}."
            )
        if self.config.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.config.beta!r}.")

    def _alignment(self, q: np.ndarray, n_families: int) -> float:
        """Specialisation (no target) or target-profile alignment, in ``[0, 1]``."""
        cfg = self.config
        if cfg.target_profile is None:
            if n_families <= 1:
                # A single axis is trivially "specialised" — nothing to be even about.
                return 1.0
            entropy = float(-np.sum(q * np.log(q + _EPS)))
            return float(np.clip(1.0 - entropy / float(np.log(n_families)), 0.0, 1.0))

        tgt = np.asarray(cfg.target_profile, dtype=np.float64).ravel()
        if tgt.size != n_families:
            raise ValueError(
                f"target_profile has {tgt.size} entries but the dataset carries "
                f"{n_families} degradation families."
            )
        if not np.all(np.isfinite(tgt)) or np.any(tgt < 0.0) or tgt.sum() <= 0.0:
            raise ValueError(
                f"target_profile must be finite, non-negative and sum to > 0; got {tgt!r}."
            )
        tgt = tgt / tgt.sum()
        dev = float(np.linalg.norm(q - tgt))
        return float(np.clip(1.0 - cfg.beta * dev, 0.0, 1.0))

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)

        cfg = self.config
        w = float(cfg.specialisation_weight)
        scores: dict[str, float] = {}
        st_per_metric: dict[str, list[float]] = {}
        coverage_per_metric: dict[str, float] = {}
        magnitude_per_metric: dict[str, float] = {}

        for name in metric_set.names():
            traj, severities, contents, families = _build_trajectory_table(dataset, name)
            family_avgs = _per_family_average(traj, families, contents)

            # Per-axis total-effect surrogate: the severity-explained variance
            # ratio. Families with no rows are dropped by ``_per_family_average``;
            # we iterate ``families`` so the axis ordering matches a supplied
            # target_profile (length n_families).
            st: list[float] = []
            for fam in families:
                avg = family_avgs.get(fam)
                if avg is None or not severities:
                    st.append(0.0)
                    continue
                st.append(severity_response_ratio(avg))
            st_per_metric[name] = st

            # Variance ratios are non-negative; a negative q would push the
            # entropy outside [0, log n] and let coverage escape [0, 1].
            st_arr = np.clip(np.asarray(st, dtype=np.float64), 0.0, None)
            total = float(st_arr.sum())
            n_families = st_arr.size

            if total <= _EPS:
                # The metric responded to NO axis, so the profile says nothing
                # about *where* it responds — by symmetry it is maximally even.
                # Dividing by max(total, _EPS) instead gives q ≈ 0, which is not
                # a distribution: its entropy collapses to 0 (so a pure-noise
                # metric reads as the LEAST even — the diagnostic inverts) and,
                # being off the simplex, is no longer bounded by log n.
                q = np.full(n_families, 1.0 / n_families) if n_families else st_arr
            else:
                q = st_arr / total

            nz = q > 0.0
            entropy = float(-np.sum(q[nz] * np.log(q[nz])))
            max_entropy = float(np.log(n_families)) if n_families > 1 else 0.0
            coverage_per_metric[name] = entropy / max_entropy if max_entropy > _EPS else 0.0

            magnitude = float(np.clip(st_arr.max(initial=0.0), 0.0, 1.0))
            magnitude_per_metric[name] = magnitude
            alignment = self._alignment(q, n_families)
            scores[name] = magnitude * ((1.0 - w) + w * alignment)

        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "total_effect": st_per_metric,
                "magnitude": magnitude_per_metric,
                "coverage": coverage_per_metric,
                "beta": cfg.beta,
                "target_profile": cfg.target_profile,
                "specialisation_weight": w,
                # The Saltelli S_T - S gap needs a factorial design; a severity
                # sweep activates one family per sample, so it is NOT estimable
                # here. Declared rather than fabricated as 0.
                "interaction_gap_estimable": False,
            },
        )


__all__ = ["SobolConfig", "SobolRanker", "severity_response_ratio"]
