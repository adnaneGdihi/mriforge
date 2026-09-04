r"""Shared statistics for reporting figures.

Bootstrap CIs, paired-bootstrap p-values, and Holm–Bonferroni correction.
Deterministic (seeded) so figures are reproducible — note ``Math.random``
analogues are avoided; all randomness flows through an explicit seed.
"""

from __future__ import annotations

import numpy as np


def t_ci_half_width(values, *, confidence: float = 0.95) -> float:
    r"""Half-width of the Student-t confidence interval for the mean.

    Returns ``t_{(1+confidence)/2, n-1} · s / sqrt(n)`` with the sample std
    (``ddof=1``). The normal-approximation factor ``z = 1.96`` is only valid for
    large ``n``; at the seed counts used in this repo it is dramatically too
    narrow (``t = 12.71`` at ``n = 2``, ``4.30`` at ``n = 3``, ``2.78`` at
    ``n = 5``), which inverts "is this component necessary?" conclusions read
    off the error bars. Returns ``0.0`` for ``n < 2`` (the interval is
    undefined for a single observation).
    """
    from scipy import stats

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = arr.size
    if n < 2:
        return 0.0
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.5 * (1.0 + confidence), df=n - 1))
    return float(t_crit * sem)


def bootstrap_ci(
    x, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def paired_bootstrap_pvalue(a, b, *, n_boot: int = 5000, seed: int = 0) -> float:
    """Two-sided paired-bootstrap p-value for mean(b) - mean(a) == 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    d = b[mask] - a[mask]
    if d.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    centred = d - d.mean()
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boot = centred[idx].mean(axis=1)
    obs = d.mean()
    p = float((np.abs(boot) >= abs(obs)).mean())
    return min(1.0, max(1.0 / n_boot, p))


def holm_bonferroni(pvalues) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()
