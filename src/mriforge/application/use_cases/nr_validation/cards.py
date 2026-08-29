"""Per-metric validation result types + Tier-1 concordance scorer.

The shared contract every tier scorer targets: each returns
``dict[metric_name -> TierResult]``; the orchestrator folds them into one
:class:`MetricCard` per metric. Definition-of-done (spec §11): a metric passes
when it clears the tiers that were run (concordance rho_S, FR-proxy, CHO, ICC,
zero invariance violations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from mriforge.core.metrics.meta_evaluation.rankers.sim2rank import _spearman_rho
from mriforge.core.metrics.meta_evaluation.types import MetricEvaluationDataset

_METRIC_CACHE: dict[str, Any] = {}


def eval_metric(name: str, pred: torch.Tensor, ref: torch.Tensor) -> float:
    """Evaluate one registered NR metric on ``pred`` with a synthesised context.

    The shared per-image evaluator for Tiers 3/4. For context-needing metrics it
    builds a :class:`MetricContext` from the clean reference's forward model
    (reusing the meta-evaluation simulator helper), so a degraded ``pred``'s
    measurement inconsistency is detectable. Never raises — returns NaN on any
    failure (a tier records NaN, not a crash).
    """
    from mriforge.core.metrics import get_metric
    from mriforge.core.metrics.registry import MetricsRegistry

    metric = _METRIC_CACHE.get(name)
    if metric is None:
        try:
            metric = get_metric(name)
        except Exception:
            return float("nan")
        _METRIC_CACHE[name] = metric
    try:
        if MetricsRegistry.needs_context(name):
            from mriforge.core.metrics.meta_evaluation.simulator import (
                _build_sample_context,
            )

            ctx = _build_sample_context(ref, MetricsRegistry.needs(name))
            return float(metric(pred, ref, context=ctx))
        return float(metric(pred, ref))
    except Exception:
        return float("nan")


def spearman(x: list[float] | torch.Tensor, y: list[float] | torch.Tensor) -> float:
    """NaN-safe Spearman rho (reuses the meta-evaluation primitive)."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    return _spearman_rho(xt, yt)


def kendall_tau(x: list[float], y: list[float]) -> float:
    """O(T²) Kendall τ; NaN-safe."""
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    mask = torch.isfinite(xt) & torch.isfinite(yt)
    if int(mask.sum().item()) < 3:
        return 0.0
    xm, ym = xt[mask], yt[mask]
    sx = torch.sign(xm.unsqueeze(0) - xm.unsqueeze(1))
    sy = torch.sign(ym.unsqueeze(0) - ym.unsqueeze(1))
    tri = torch.triu(torch.ones_like(sx), diagonal=1).bool()
    n = float(tri.sum().item())
    if n < 1:
        return 0.0
    return float((sx * sy)[tri].sum().item() / n)


@dataclass
class TierResult:
    """One tier's verdict for one metric."""

    tier: str
    passed: bool
    score: float  # the tier's headline statistic
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "passed": bool(self.passed),
            "score": float(self.score) if self.score == self.score else None,
            "detail": self.detail,
        }


@dataclass
class MetricCard:
    """All tier verdicts for one metric + the overall pass/fail."""

    metric: str
    tiers: dict[str, TierResult] = field(default_factory=dict)

    @property
    def overall_pass(self) -> bool:
        """Pass iff every *run* tier passed (a metric not scored on a tier is
        neither pass nor fail there — overall is over the tiers present)."""
        return bool(self.tiers) and all(t.passed for t in self.tiers.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "overall_pass": self.overall_pass,
            "tiers": {k: v.to_dict() for k, v in self.tiers.items()},
        }


def build_cards(
    per_tier: dict[str, dict[str, TierResult]],
) -> dict[str, MetricCard]:
    """Fold ``{tier_name: {metric: TierResult}}`` into ``{metric: MetricCard}``."""
    cards: dict[str, MetricCard] = {}
    for tier_name, results in per_tier.items():
        for metric, res in results.items():
            cards.setdefault(metric, MetricCard(metric=metric)).tiers[tier_name] = res
    return cards


def score_concordance(
    dataset: MetricEvaluationDataset,
    metric_names: list[str],
    *,
    rho_threshold: float = 0.6,
) -> dict[str, TierResult]:
    """§8.1 Tier 1 — physics-anchored concordance, per metric.

    For each metric: Spearman rho_S and Kendall τ_K between the metric trajectory
    and the degradation severity θ, averaged across degradation families, plus a
    strict-monotonicity flag (the metric is monotone in θ within each family).
    Pass: ``|rho_S| ≥ rho_threshold`` AND monotone in at least half the families.
    Direction-agnostic (uses ``|rho_S|``) — a metric that tracks severity with
    either sign is concordant; the registry ``higher_is_better`` orients it
    downstream.
    """
    samples = dataset.samples
    families = sorted({s.degradation_family for s in samples})
    # index samples by family → (theta, value) lists
    out: dict[str, TierResult] = {}
    for name in metric_names:
        values = dataset.metric_values.get(name)
        if values is None:
            out[name] = TierResult(
                "concordance", False, float("nan"), {"reason": "metric not in dataset"}
            )
            continue
        rhos: list[float] = []
        taus: list[float] = []
        monotone_flags: list[bool] = []
        for fam in families:
            idx = [i for i, s in enumerate(samples) if s.degradation_family == fam]
            thetas = [float(samples[i].severity.get("theta", 0.0)) for i in idx]
            vals = [float(values[i]) for i in idx]
            finite = [(t, v) for t, v in zip(thetas, vals, strict=True) if v == v]
            if len(finite) < 3:
                continue
            ts, vs = zip(*sorted(finite), strict=True)
            # Guard a constant trajectory: the repo's tie-blind Spearman returns
            # a spurious rho=1 on constant input (argsort assigns sequential ranks
            # in theta-sorted order). A flat metric has zero discrimination, rho=0.
            rho = 0.0 if max(vs) - min(vs) < 1e-12 else spearman(list(vs), list(ts))
            rhos.append(rho)
            taus.append(kendall_tau(list(vs), list(ts)))
            # strict monotonicity: sign-consistent first differences
            diffs = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
            nz = [d for d in diffs if abs(d) > 1e-12]
            monotone_flags.append(bool(nz) and (all(d > 0 for d in nz) or all(d < 0 for d in nz)))
        if not rhos:
            out[name] = TierResult(
                "concordance", False, float("nan"), {"reason": "insufficient finite samples"}
            )
            continue
        mean_abs_rho = float(sum(abs(r) for r in rhos) / len(rhos))
        mean_tau = float(sum(abs(t) for t in taus) / len(taus)) if taus else 0.0
        monotone_frac = sum(monotone_flags) / len(monotone_flags)
        passed = mean_abs_rho >= rho_threshold and monotone_frac >= 0.5
        out[name] = TierResult(
            "concordance",
            passed,
            mean_abs_rho,
            {
                "mean_abs_rho_s": mean_abs_rho,
                "mean_abs_tau_k": mean_tau,
                "monotone_fraction": monotone_frac,
                "n_families": len(rhos),
            },
        )
    return out


__all__ = [
    "MetricCard",
    "TierResult",
    "build_cards",
    "eval_metric",
    "kendall_tau",
    "score_concordance",
    "spearman",
]
