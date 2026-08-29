"""Gen-2 MS³ ranker — Monotonicity-Sensitivity-Specificity Sweep.

In-core, ``BaseRanker``-conformant port of ``compute_ms3_score`` from
``scripts/sim2rank/meta_eval.py``. Three trajectory statistics — per-axis
polarity-oriented Spearman monotonicity, a normalised dynamic range
(sensitivity), and a shape-only cross-axis contrast (specificity) — fold into one
aggregate per metric:

.. math::
    \\mathcal{R}_k^{(\\mathrm{MS^3})}
        = \\sum_j w_j\\,\\mathrm{SROCC}^{+}_{kj}
        + \\lambda_1 \\sum_j w_j\\,\\mathcal{S}_{kj}
        + \\lambda_2\\,\\mathcal{C}_k
        \\;\\in\\; [0,\\; 1 + \\lambda_1 + \\lambda_2],

with uniform axis weights :math:`w_j = 1/|families|` and
:math:`\\lambda_1 = \\lambda_2 = 0.3`. Axes are the degradation *families*; the
per-axis row is the content-averaged trajectory along the ascending severity grid
(the same ``_per_family_average`` table the Gen-1 rankers use).

**Every term is dimensionless and bounded in** ``[0, 1]`` (issue #248). Before the
2026-07 correctness sweep they were not, and the ranker was inverted:

* *Specificity* was an **offset** contrast, ``mean_t|m_a − m_b| / √(½(Var_a +
  Var_b))`` — unbounded, and added raw to a SROCC term bounded in ``[0, 1]``. A
  pure-noise metric carrying a constant ``+5.0`` offset on one axis scored an
  aggregate of **23.7 against 1.98** for a perfectly monotone metric. It is now
  ``√(SROCC_a·SROCC_b)·(1 − |corr(m_a, m_b)|)``: correlation centres each
  trajectory (so an offset contributes nothing), and the ``√(SROCC·SROCC)`` factor
  gates the contrast on the metric actually *responding* — without that gate,
  plain decorrelation hands noise, which is uncorrelated across axes by
  construction, a specificity of ≈1.
* *Sensitivity* was ``mean_t|Δm/Δs| / σ_m``, carrying units of ``1/severity``:
  relabelling severity from a fraction to a percent **flipped the winner**. It is
  now the normalised dynamic range ``|m(s_T) − m(s_1)| / (σ_m·√(2T))``, in which
  ``Δs`` does not appear at all. (Merely range-normalising ``Δs`` — the minimal
  repair — fixes the units but leaves a *roughness* statistic that white noise
  maximises: it grows like ``1.128·(T−1)`` for noise against ``≈3.4`` for a smooth
  monotone response, so at ``T ≥ 6`` noise wins the term outright.)
* ``higher_is_better`` was a **required argument that nothing read**. It now
  orients the monotonicity term: a quality metric declared higher-is-better must
  *fall* as severity rises, so ``SROCC⁺ = max(0, ∓ρ)`` and a metric that tracks
  degradation backwards scores 0 instead of the perfect ``|ρ| = 1`` it used to
  collect. ``diagnostics["srocc"]`` keeps the unsigned ``|ρ|`` (the Sobol
  total-effect surrogate and the cross-axis figure want the magnitude).
* The legacy ``plcc_per_axis`` (a 4-parameter logistic fit per metric per axis)
  never entered the aggregate and had no consumer — it is gone from both sides.

The SROCC term uses :func:`srocc_signed` (scipy mid-rank ``spearmanr`` + the
``_spearman_per_metric`` all-NaN / std<1e-12 → 0 guard), so it stays faithful to
the legacy on tied AND constant trajectories.

NumPy ``var``/``std`` are population statistics (``ddof=0``), so every torch moment
here is computed with ``unbiased=False`` to preserve parity with the legacy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from ._spearman_legacy import srocc_signed
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import (
    _build_trajectory_table,
    _per_family_average,
)


@dataclass
class MS3Config:
    """MS³ aggregation weights (mirror the legacy ``compute_ms3_score`` defaults)."""

    lambda_sens: float = 0.3  # λ₁ — weight on the sensitivity term
    lambda_spec: float = 0.3  # λ₂ — weight on the specificity term


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


def _abs_pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """``|corr(a, b)|`` over the severity axis; constant input → 0.0, never NaN."""
    x = np.nan_to_num(a.detach().cpu().numpy().astype(np.float64))
    y = np.nan_to_num(b.detach().cpu().numpy().astype(np.float64))
    if x.size < 2 or x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    return 0.0 if not np.isfinite(r) else min(abs(r), 1.0)


@register_ranker(
    "ms3",
    generation=2,
    description="Monotonicity-Sensitivity-Specificity Sweep (MS3)",
)
class MS3Ranker(BaseRanker):
    """Monotonicity-Sensitivity-Specificity Sweep ranker (Gen-2)."""

    method_name = "MS3"

    def __init__(self, config: MS3Config | None = None) -> None:
        self.config = config or MS3Config()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)

        names = metric_set.names()
        n_metrics = len(names)

        # The family set is shared across metrics (same dataset) — derive the
        # severity grid + family list once from any metric's trajectory table.
        _traj0, severities0, _contents0, families0 = _build_trajectory_table(dataset, names[0])
        severity_grid = torch.tensor(severities0, dtype=torch.float64)
        families = families0

        # mat[fam] is the (M, T) per-axis trajectory matrix: row j = metric
        # ``names[j]``'s content-averaged, severity-ascending trajectory for the
        # degradation family ``fam``. This mirrors the legacy ``axis_trajectories``.
        mat: dict[str, torch.Tensor] = {}
        for name in names:
            traj, _sev, contents, _fams = _build_trajectory_table(dataset, name)
            family_avgs = _per_family_average(traj, families, contents)
            for fam in families:
                row = family_avgs.get(fam)
                if row is None:
                    continue
                mat.setdefault(fam, []).append(row)  # type: ignore[union-attr]
        mat = {
            fam: torch.stack(rows, dim=0)  # type: ignore[arg-type]
            for fam, rows in mat.items()
            if rows
        }
        axes = list(mat.keys())

        if not axes:
            scores = dict.fromkeys(names, 0.0)
            return RankingResult(
                method=self.method_name,
                scores=scores,
                ranks=order_from_scores(scores),
                diagnostics={
                    "axes": [],
                    "lambda_sens": self.config.lambda_sens,
                    "lambda_spec": self.config.lambda_spec,
                },
            )

        # Uniform axis weights, renormalised to sum to 1 (legacy default).
        weights = {a: 1.0 / len(axes) for a in axes}
        wsum = sum(weights.values())
        weights = {k: v / wsum for k, v in weights.items()}

        # A higher-is-better metric must DECREASE as severity rises; a
        # lower-is-better metric must increase. ``expected_sign`` maps the signed ρ
        # onto "tracks the degradation correctly" = positive (issue #248c).
        expected_sign = torch.tensor(
            [-1.0 if metric_set.is_higher_better(n) else 1.0 for n in names],
            dtype=torch.float64,
        )

        srocc: dict[str, torch.Tensor] = {}  # |ρ| — magnitude (Sobol/figure input)
        srocc_oriented: dict[str, torch.Tensor] = {}  # max(0, ∓ρ) — the score term
        sens: dict[str, torch.Tensor] = {}

        n_t = int(severity_grid.numel())
        sqrt_2t = math.sqrt(2.0 * max(n_t, 1))

        for axis in axes:
            traj = mat[axis]  # (M, T)

            # Signed tie-corrected Spearman per metric row (scipy mid-ranks + the
            # exact ``_spearman_per_metric`` all-NaN / population-std<1e-12 → 0
            # guard). The in-core tie-blind ``_spearman_rho`` diverges on
            # tied/constant rows — adversarial verification 2026-05-30.
            rho_rows = torch.tensor(
                [srocc_signed(traj[i], severity_grid) for i in range(n_metrics)],
                dtype=torch.float64,
            )
            srocc[axis] = rho_rows.abs()
            srocc_oriented[axis] = (expected_sign * rho_rows).clamp(0.0, 1.0)

            # Sensitivity — normalised dynamic range, dimensionless and in [0, 1].
            # √(2T) is the supremum of |m(s_T) − m(s_1)| / σ_m over all length-T
            # trajectories (attained by a ±spike at the two endpoints).
            filled = torch.nan_to_num(traj)
            if n_t >= 2:
                net = (filled[:, -1] - filled[:, 0]).abs()
            else:
                net = torch.zeros(n_metrics, dtype=torch.float64)
            sigma_m = filled.std(dim=1, unbiased=False)
            sigma_m = torch.where(
                sigma_m > 1e-12,
                sigma_m,
                torch.full_like(sigma_m, float("inf")),  # constant row → sens 0
            )
            sens[axis] = (net / (sigma_m * sqrt_2t)).clamp(0.0, 1.0)

        # Specificity — shape-only, responsiveness-gated, bounded in [0, 1]:
        # mean over ordered (a, b), a != b, of √(SROCC_a·SROCC_b)·(1 − |corr(a, b)|).
        spec = torch.zeros(n_metrics, dtype=torch.float64)
        n_pairs = 0
        for a in axes:
            for b in axes:
                if a == b:
                    continue
                gate = torch.sqrt(srocc[a] * srocc[b])  # both axes must respond
                shape = torch.tensor(
                    [1.0 - _abs_pearson(mat[a][i], mat[b][i]) for i in range(n_metrics)],
                    dtype=torch.float64,
                )
                spec += gate * shape
                n_pairs += 1
        if n_pairs > 0:
            spec /= n_pairs

        # Aggregate: Σ_j w_j SROCC⁺_kj + λ1 Σ_j w_j Sens_kj + λ2 Spec_k.
        srocc_term = torch.zeros(n_metrics, dtype=torch.float64)
        sens_term = torch.zeros(n_metrics, dtype=torch.float64)
        for axis, w in weights.items():
            srocc_term += w * srocc_oriented[axis]
            sens_term += w * sens[axis]

        aggregate = (
            srocc_term + self.config.lambda_sens * sens_term + self.config.lambda_spec * spec
        )

        scores = {name: float(aggregate[j].item()) for j, name in enumerate(names)}
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "axes": axes,
                "srocc": {a: srocc[a].tolist() for a in axes},
                "srocc_oriented": {a: srocc_oriented[a].tolist() for a in axes},
                "sensitivity": {a: sens[a].tolist() for a in axes},
                "specificity": spec.tolist(),
                "weights": weights,
                "lambda_sens": self.config.lambda_sens,
                "lambda_spec": self.config.lambda_spec,
            },
        )


__all__ = ["MS3Config", "MS3Ranker"]
