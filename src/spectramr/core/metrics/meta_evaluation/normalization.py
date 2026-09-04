"""Cross-metric normalization schemes for the meta-evaluation report.

The report mixes metrics on incompatible scales — PSNR (~0-50 dB),
FID (0-400+), LPIPS/SSIM ([0, 1]), MSE ([0, inf)), g-factor ([1, inf)).
A single leaderboard table or figure is only legible if every metric is
projected onto a comparable axis first. These functions take one scalar
per metric (e.g. a clean-baseline mean, or the final consensus z-score)
and return a comparable scalar, *without mutating the input*.

Three schemes, all NaN-safe and direction-aware:

``direction_aware_minmax`` (default)
    Flip lower-is-better metrics, then min-max the cohort to ``[0, 1]`` so
    ``1.0`` = best observed, ``0.0`` = worst observed. Intuitive for a
    non-specialist reader, bounded, unit-free. A constant cohort maps to
    ``0.5``. This is the headline scheme for the leaderboard.

``robust_zscore``
    Center on the median, scale by the MAD (x1.4826 for normal
    consistency). Resists heavy tails (FID/MSE outliers) better than a
    mean/std z-score, but is unbounded and less intuitive at a glance.

``percentile_rank``
    Replace each value by its within-cohort percentile in ``[0, 1]``.
    Fully scale-free and outlier-proof, but discards magnitude.

The ``higher_is_better`` argument is the per-metric optimization direction
(the SSOT lives in :data:`spectramr.core.metrics.metric_directions.
METRIC_HIGHER_IS_BETTER`). When it is ``None`` every metric is treated as
larger-is-better — correct for consensus z-scores, which are already
direction-normalized upstream.
"""

from __future__ import annotations

import math
from typing import Literal

NormalizationScheme = Literal["minmax", "robust_z", "percentile"]

_MAD_SCALE = 1.4826  # MAD -> sigma for a normal distribution


def _signed_finite(
    values: dict[str, float],
    higher_is_better: dict[str, bool] | None,
) -> tuple[dict[str, float], list[float]]:
    """Apply the direction flip and split out the finite values.

    Returns ``(signed, finite_values)`` where ``signed`` maps every metric
    to its direction-corrected value (NaN preserved) and ``finite_values``
    is the list of finite signed values used to compute cohort statistics.
    """
    signed: dict[str, float] = {}
    finite: list[float] = []
    for name, raw in values.items():
        v = float(raw)
        if higher_is_better is not None and not higher_is_better.get(name, True):
            v = -v
        signed[name] = v
        if math.isfinite(v):
            finite.append(v)
    return signed, finite


def direction_aware_minmax(
    values: dict[str, float],
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Min-max each metric to ``[0, 1]`` after a direction flip (best = 1)."""
    if not values:
        return {}
    signed, finite = _signed_finite(values, higher_is_better)
    if not finite:
        return dict(signed)  # all-NaN cohort: nothing to scale
    lo, hi = min(finite), max(finite)
    span = hi - lo
    if span < 1e-12:
        return {k: (float("nan") if math.isnan(v) else 0.5) for k, v in signed.items()}
    return {k: (float("nan") if math.isnan(v) else (v - lo) / span) for k, v in signed.items()}


def robust_zscore(
    values: dict[str, float],
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Median/MAD z-score after a direction flip (heavy-tail-robust)."""
    if not values:
        return {}
    signed, finite = _signed_finite(values, higher_is_better)
    if not finite:
        return dict(signed)
    median = _median(finite)
    mad = _median([abs(v - median) for v in finite])
    scale = mad * _MAD_SCALE
    if scale < 1e-12:
        return {k: (float("nan") if math.isnan(v) else 0.0) for k, v in signed.items()}
    return {k: (float("nan") if math.isnan(v) else (v - median) / scale) for k, v in signed.items()}


def percentile_rank(
    values: dict[str, float],
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Within-cohort percentile in ``[0, 1]`` after a direction flip."""
    if not values:
        return {}
    signed, finite = _signed_finite(values, higher_is_better)
    if len(finite) <= 1:
        # A single finite value has no spread to rank against.
        return {k: (float("nan") if math.isnan(v) else 0.5) for k, v in signed.items()}
    ordered = sorted(finite)
    n = len(ordered)
    out: dict[str, float] = {}
    for k, v in signed.items():
        if math.isnan(v):
            out[k] = float("nan")
            continue
        # Fraction of finite values strictly below v -> [0, 1].
        below = sum(1 for x in ordered if x < v)
        out[k] = below / (n - 1)
    return out


def normalize(
    values: dict[str, float],
    scheme: NormalizationScheme,
    higher_is_better: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Dispatch to a named scheme. Raises on an unknown scheme (pitfall #9)."""
    dispatch = {
        "minmax": direction_aware_minmax,
        "robust_z": robust_zscore,
        "percentile": percentile_rank,
    }
    fn = dispatch.get(scheme)
    if fn is None:
        raise ValueError(
            f"unknown normalization scheme {scheme!r}; choose one of {sorted(dispatch)}"
        )
    return fn(values, higher_is_better)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
