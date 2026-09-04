"""§8.2 Tier 2 — full-reference (FR) proxy agreement, per NR metric.

An NR metric that does not track *any* established full-reference quality metric
on the shared degradation sweep is not measuring image quality. This tier scores
each NR metric by its best (maximum-absolute) Spearman agreement with the FR
proxy columns the orchestrator already placed in
:class:`~spectramr.core.metrics.meta_evaluation.types.MetricEvaluationDataset`.

It also computes the **incremental-validity** diagnostic (variance inflation
factors): an NR metric whose column is well predicted by a linear combination of
the *other* NR columns is redundant. ``VIF_j = 1 / (1 - R_j²)`` (spec). High VIF
(>~10) flags redundancy and is folded into each metric's
:class:`~spectramr.application.use_cases.nr_validation.cards.TierResult` detail.

The scorer is CPU-canonical, deterministic, and never raises — each statistic is
wrapped in ``try/except`` and records ``float('nan')`` on failure (mirroring the
crash→NaN contract of ``precompute_metric_values``).
"""

from __future__ import annotations

import math
from typing import Any

import torch

from spectramr.application.use_cases.nr_validation.cards import TierResult, spearman
from spectramr.core.metrics.meta_evaluation.types import MetricEvaluationDataset

__all__ = [
    "DEFAULT_FR_PROXIES",
    "score_fr_proxy",
    "variance_inflation_factors",
]

# Established full-reference proxies, default order = preference for tie-breaking
# the headline ``best_fr`` label. These columns come straight off the dataset's
# ``metric_values`` (the orchestrator computes them) — they are NEVER recomputed
# here.
DEFAULT_FR_PROXIES: list[str] = ["psnr", "ssim", "ms_ssim", "lpips", "dists"]

# Cap returned for VIF when R² → 1 (perfectly collinear column): VIF diverges, so
# we report a large finite sentinel rather than +inf so the value stays JSON-safe
# and orderable.
_VIF_CAP: float = 1.0e6


def _is_constant(col: list[float], *, tol: float = 1e-12) -> bool:
    """True when a column's finite entries span less than ``tol`` (no variance).

    A constant metric has no discrimination, so it must not earn a spurious
    rank correlation from the repo's tie-blind Spearman. Never raises.
    """
    try:
        t = torch.as_tensor(col, dtype=torch.float64)
        finite = t[torch.isfinite(t)]
        if finite.numel() < 2:
            return True
        return bool((finite.max() - finite.min()).item() < tol)
    except Exception:
        return False


def _n_finite_pairs(a: list[float], b: list[float]) -> int:
    """Number of index positions finite in both ``a`` and ``b``. Never raises."""
    try:
        at = torch.as_tensor(a, dtype=torch.float64)
        bt = torch.as_tensor(b, dtype=torch.float64)
        return int((torch.isfinite(at) & torch.isfinite(bt)).sum().item())
    except Exception:
        return 0


def variance_inflation_factors(
    dataset: MetricEvaluationDataset,
    nr_names: list[str],
) -> dict[str, float]:
    """Variance inflation factor of each NR metric against the other NR metrics.

    For each NR metric ``j`` we regress its column on *all other* NR columns
    (intercept included) using a dependency-free ordinary-least-squares solve
    (:func:`torch.linalg.lstsq` on the ``float64`` design matrix). With
    ``R_j² = 1 - SSE / SST`` the variance inflation factor is

    .. math:: \\mathrm{VIF}_j = \\frac{1}{1 - R_j^2}.

    A high VIF (>~10) flags a redundant metric whose information is already
    carried by the rest of the battery. Only rows finite across *all* involved
    columns are used; if fewer than three such rows remain (or fewer than two NR
    metrics are supplied) the VIF is ``nan``. As ``R_j² → 1`` the VIF is capped
    at a large finite sentinel. Never raises.

    Args:
        dataset: The shared degradation-sweep dataset carrying NR metric columns.
        nr_names: The NR metric names to compute mutual VIFs for.

    Returns:
        Mapping ``nr_name -> VIF`` (``float``; ``nan`` when undetermined).
    """
    out: dict[str, float] = {}
    present = [n for n in nr_names if n in dataset.metric_values]
    # VIF needs at least one regressor, i.e. >= 2 NR columns.
    if len(present) < 2:
        return {n: float("nan") for n in nr_names}

    try:
        cols = {n: torch.as_tensor(dataset.metric_values[n], dtype=torch.float64) for n in present}
    except Exception:
        return {n: float("nan") for n in nr_names}

    for n in nr_names:
        if n not in cols:
            out[n] = float("nan")
            continue
        others = [m for m in present if m != n]
        if not others:
            out[n] = float("nan")
            continue
        try:
            out[n] = _single_vif(cols[n], [cols[m] for m in others])
        except Exception:
            out[n] = float("nan")
    return out


def _single_vif(target: torch.Tensor, regressors: list[torch.Tensor]) -> float:
    """OLS R² → VIF for one target column against a list of regressor columns.

    Rows non-finite in any involved column are dropped. Returns ``nan`` when
    fewer than three usable rows remain, and a large finite cap when the target
    is (numerically) constant or perfectly explained.
    """
    y = target.to(torch.float64)
    xs = [r.to(torch.float64) for r in regressors]
    finite = torch.isfinite(y)
    for r in xs:
        finite = finite & torch.isfinite(r)
    if int(finite.sum().item()) < 3:
        return float("nan")

    y = y[finite]
    n_rows = y.shape[0]
    sst = float(((y - y.mean()) ** 2).sum().item())
    # A constant target has no variance to explain — VIF is undefined; treat as
    # maximally inflated (it is perfectly "predicted" by the intercept alone).
    if sst < 1e-24:
        return _VIF_CAP

    design = torch.stack(
        [torch.ones(n_rows, dtype=torch.float64)] + [r[finite] for r in xs],
        dim=1,
    )
    sol = torch.linalg.lstsq(design, y.unsqueeze(1)).solution
    pred = (design @ sol).squeeze(1)
    sse = float(((y - pred) ** 2).sum().item())
    r2 = 1.0 - sse / sst
    # Clamp numerical overshoot/undershoot into [0, 1].
    r2 = min(1.0, max(0.0, r2))
    denom = 1.0 - r2
    if denom < 1.0 / _VIF_CAP:
        return _VIF_CAP
    return float(1.0 / denom)


def score_fr_proxy(
    dataset: MetricEvaluationDataset,
    nr_names: list[str],
    fr_names: list[str] | None = None,
    *,
    rho_threshold: float = 0.6,
) -> dict[str, TierResult]:
    """§8.2 Tier 2 — agreement of each NR metric with full-reference proxies.

    For every NR metric the scorer computes the absolute Spearman correlation
    ``|rho_S|`` against each FR proxy column present in the dataset (over the
    finite-paired entries), using :func:`cards.spearman`. The FR columns are
    read straight off ``dataset.metric_values`` and are **never recomputed**.
    The headline score is the maximum ``|rho_S|`` across the available proxies
    (best agreement); the metric passes when that best agreement is at least
    ``rho_threshold``. The mutual variance inflation factor (see
    :func:`variance_inflation_factors`) is folded into each detail under
    ``"vif"`` so the metric card carries the redundancy diagnostic.

    Args:
        dataset: The shared degradation-sweep dataset (NR + FR columns).
        nr_names: NR metric names to score.
        fr_names: FR proxy names to correlate against. Defaults to
            :data:`DEFAULT_FR_PROXIES`. Only those actually present in the
            dataset are used.
        rho_threshold: Minimum best ``|rho_S|`` for a pass.

    Returns:
        Mapping ``nr_name -> TierResult`` (tier ``"fr_proxy"``). When no FR proxy
        column is present in the dataset every result is a failure carrying
        ``detail={"reason": "no FR proxy columns in dataset"}``. Never raises.
    """
    if fr_names is None:
        fr_names = list(DEFAULT_FR_PROXIES)

    available_fr = [f for f in fr_names if f in dataset.metric_values]
    vifs = variance_inflation_factors(dataset, nr_names)

    out: dict[str, TierResult] = {}

    if not available_fr:
        for name in nr_names:
            out[name] = TierResult(
                "fr_proxy",
                False,
                float("nan"),
                {"reason": "no FR proxy columns in dataset", "vif": vifs.get(name, float("nan"))},
            )
        return out

    for name in nr_names:
        vif = vifs.get(name, float("nan"))
        nr_col = dataset.metric_values.get(name)
        if nr_col is None:
            out[name] = TierResult(
                "fr_proxy",
                False,
                float("nan"),
                {"reason": "metric not in dataset", "vif": vif},
            )
            continue

        per_fr: dict[str, float] = {}
        for fr in available_fr:
            try:
                fr_col = dataset.metric_values[fr]
                # <3 finite-paired rows ⇒ correlation is undetermined ⇒ NaN
                # (mirrors the spec edge case and ``score_concordance``).
                if _n_finite_pairs(nr_col, fr_col) < 3:
                    per_fr[fr] = float("nan")
                    continue
                # A flat column has zero discrimination. The repo's tie-blind
                # Spearman returns a spurious |rho|≈1 on a constant input (argsort
                # assigns sequential ranks), so guard it — a constant metric
                # correlates with nothing.
                if _is_constant(nr_col) or _is_constant(fr_col):
                    per_fr[fr] = 0.0
                    continue
                rho = spearman(nr_col, fr_col)
                per_fr[fr] = abs(rho) if rho == rho else float("nan")
            except Exception:
                per_fr[fr] = float("nan")

        finite_fr = {k: v for k, v in per_fr.items() if math.isfinite(v)}
        if not finite_fr:
            out[name] = TierResult(
                "fr_proxy",
                False,
                float("nan"),
                {
                    "per_fr": per_fr,
                    "best_fr": None,
                    "n_fr": len(available_fr),
                    "vif": vif,
                    "reason": "no finite NR↔FR correlation",
                },
            )
            continue

        best_fr = max(finite_fr, key=lambda k: finite_fr[k])
        score = finite_fr[best_fr]
        passed = bool(score >= rho_threshold)
        detail: dict[str, Any] = {
            "per_fr": per_fr,
            "best_fr": best_fr,
            "n_fr": len(available_fr),
            "vif": vif,
        }
        out[name] = TierResult("fr_proxy", passed, float(score), detail)

    return out
