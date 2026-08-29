r"""Cross-site conformal aggregator (idea 8 §8.2).

Combines per-site calibration quantiles
:math:`\hat q_{1-\alpha}^{(k)}` into a single conformal threshold
under the *quantile-only* federation protocol (Barber et al.
*Ann. Stat.* 2023). Sites share their quantile but never their raw
calibration scores — preserves the privacy guarantee that is the
whole point of the federated DP-conformal pipeline.

Two aggregation rules are supported:

- ``"max"``: ``q_global = max_k q_k``. Conservative — gives the
  strongest worst-case coverage but the widest band.
- ``"weighted"``: ``q_global = sum_k (n_k / N) q_k`` with ``n_k``
  the per-site calibration sample count and ``N = sum n_k``. Tight
  bound when sites are exchangeable; less robust to non-iid sites.

The plan defaults to ``"max"`` for the ULF use case where sites are
explicitly *non*-exchangeable.

References
----------
- Barber, Candès, Ramdas, Tibshirani (2023). *Conformal prediction
  beyond exchangeability*. Ann. Statist. 51(2): 816–845.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteCalibration:
    """Per-site calibration summary shared with the aggregator.

    Attributes:
        site_id: Short identifier; only used for diagnostics.
        quantile: ``hat q_{1-α}^{(k)}`` for this site.
        n_calibration: Calibration-set size used to compute it.
    """

    site_id: str
    quantile: float
    n_calibration: int


@dataclass(frozen=True)
class AggregationReport:
    """Output of :func:`aggregate_site_quantiles`.

    Attributes:
        global_quantile: The chosen federated band radius.
        rule: Aggregation rule used (``"max"`` or ``"weighted"``).
        contributing_sites: Number of sites that contributed.
        weights: Per-site weight applied (uniform for ``"max"``;
            ``n_k / N`` for ``"weighted"``).
    """

    global_quantile: float
    rule: str
    contributing_sites: int
    weights: tuple[float, ...]


def aggregate_site_quantiles(
    sites: Sequence[SiteCalibration],
    rule: str = "max",
) -> AggregationReport:
    """Combine per-site calibration quantiles into a federated threshold.

    Args:
        sites: One :class:`SiteCalibration` per participating site.
            Must be non-empty.
        rule: ``"max"`` (Barber-2023 worst-case) or ``"weighted"``
            (sample-size-weighted).

    Returns:
        :class:`AggregationReport`.

    Raises:
        ValueError: empty input, unknown rule, or any per-site
            ``n_calibration <= 0``.
    """
    if not sites:
        raise ValueError("sites is empty")
    if rule not in ("max", "weighted"):
        raise ValueError(f"rule must be 'max' or 'weighted'; got {rule!r}")
    for s in sites:
        if s.n_calibration <= 0:
            raise ValueError(
                f"Site {s.site_id!r} has n_calibration={s.n_calibration}; must be > 0."
            )
        if s.quantile < 0:
            raise ValueError(f"Site {s.site_id!r} has negative quantile={s.quantile}.")

    if rule == "max":
        weights = tuple(1.0 / len(sites) for _ in sites)
        global_q = max(s.quantile for s in sites)
    else:
        total_n = sum(s.n_calibration for s in sites)
        weights = tuple(s.n_calibration / total_n for s in sites)
        global_q = sum(w * s.quantile for w, s in zip(weights, sites))

    return AggregationReport(
        global_quantile=float(global_q),
        rule=rule,
        contributing_sites=len(sites),
        weights=weights,
    )


__all__ = [
    "AggregationReport",
    "SiteCalibration",
    "aggregate_site_quantiles",
]
