"""Pool hygiene — clone and broken-metric screening (critique §8 repair).

Both cross-axis aggregators are pool-composition-fragile:

* **Clone inflation.** Adding ``C`` near-duplicate metrics multiplies that
  cluster's effective weight in any aggregator that treats metrics symmetrically,
  pulling the cluster toward the top (Borda) or shifting every other metric's
  pool moments (z-standardisation).
* **Broken metrics.** A metric that is large-but-uninformative (a saturated
  ``tanh`` of an irrelevant feature, or a constant) is not filtered by the NaN
  guard, yet it inflates the min/max of every min-max normalisation and collapses
  the legitimate metrics' dynamic range.

The repair (a registry-level screen applied *before* normalisation) detects both
on the pre-computed metric trajectories:

* clones — pairs with absolute Spearman correlation ``>= clone_threshold`` over the
  evaluation samples are clustered; one canonical member per cluster votes, the
  rest stay visible in per-axis reports but are excluded from the aggregator.
* broken — metrics whose absolute Spearman correlation with a universal
  degradation proxy (residual energy) is below ``signal_threshold``, or whose
  trajectory is degenerate (near-zero variance), are excluded.

The curated ``voting_metrics`` set is what the aggregator should normalise and
combine; everything else is reported for transparency, not silently dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.stats import spearmanr

from .types import MetricEvaluationDataset


@dataclass
class PoolScreening:
    """Result of :func:`screen_pool`.

    Attributes:
        clone_clusters: groups of mutually clone-correlated metrics (size >= 2).
        canonical: one representative per clone cluster (plus every un-cloned,
            non-broken metric) — the metrics permitted to vote.
        redundant: non-canonical clone members (visible, non-voting).
        broken: degenerate / uninformative metrics (excluded from voting).
        voting_metrics: the curated pool the aggregator should use
            (``canonical`` minus ``broken``), order-preserving the input.
        signal: per-metric absolute Spearman correlation with the degradation proxy.
    """

    clone_clusters: list[list[str]]
    canonical: list[str]
    redundant: list[str]
    broken: list[str]
    voting_metrics: list[str]
    signal: dict[str, float] = field(default_factory=dict)


def _abs_spearman(a: list[float], b: list[float]) -> float:
    """Absolute Spearman correlation; 0.0 when undefined (constant input).

    Constant inputs are short-circuited to avoid scipy's ``ConstantInputWarning``
    (the correlation is genuinely undefined, and a constant metric is already
    flagged as broken by its zero variance).
    """
    if min(a) == max(a) or min(b) == max(b):
        return 0.0
    rho = spearmanr(a, b).correlation
    if rho is None or math.isnan(rho):
        return 0.0
    return abs(float(rho))


def screen_pool(
    dataset: MetricEvaluationDataset,
    *,
    clone_threshold: float = 0.95,
    signal_threshold: float = 0.1,
) -> PoolScreening:
    """Screen the metric pool for clones and broken metrics before aggregation.

    Args:
        dataset: the frozen evaluation dataset (provides per-metric trajectories
            and the residual-energy degradation proxy).
        clone_threshold: absolute Spearman correlation at/above which two metrics
            are deemed clones (default 0.95).
        signal_threshold: minimum absolute Spearman correlation with the
            degradation proxy for a metric to count as informative (default 0.1).

    Returns:
        A :class:`PoolScreening`.
    """
    names = list(dataset.metric_values.keys())
    proxy = dataset.residual_energies().tolist()

    # 1) Broken: degenerate variance OR no tracking of the degradation proxy.
    signal: dict[str, float] = {}
    broken: list[str] = []
    for name in names:
        values = dataset.metric_values[name]
        lo, hi = min(values), max(values)
        s = _abs_spearman(values, proxy)
        signal[name] = s
        if hi - lo <= 1e-12 or s < signal_threshold:
            broken.append(name)

    # 2) Clones among the informative metrics (union-find on the |rho| graph).
    live = [n for n in names if n not in broken]
    parent = {n: n for n in live}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    for i in range(len(live)):
        for j in range(i + 1, len(live)):
            a, b = live[i], live[j]
            if _abs_spearman(dataset.metric_values[a], dataset.metric_values[b]) >= clone_threshold:
                union(a, b)

    clusters: dict[str, list[str]] = {}
    for n in live:
        clusters.setdefault(find(n), []).append(n)

    clone_clusters: list[list[str]] = []
    canonical: list[str] = []
    redundant: list[str] = []
    for members in clusters.values():
        # Canonical = strongest degradation tracker (tie-break: input order).
        rep = max(members, key=lambda m: (signal[m], -names.index(m)))
        canonical.append(rep)
        if len(members) >= 2:
            clone_clusters.append(sorted(members, key=names.index))
            redundant.extend(m for m in members if m != rep)

    canonical_set = set(canonical)
    voting_metrics = [n for n in names if n in canonical_set]
    return PoolScreening(
        clone_clusters=clone_clusters,
        canonical=sorted(canonical, key=names.index),
        redundant=sorted(redundant, key=names.index),
        broken=broken,
        voting_metrics=voting_metrics,
        signal=signal,
    )
