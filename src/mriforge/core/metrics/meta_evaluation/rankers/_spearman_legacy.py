"""Legacy-faithful per-metric Spearman SROCC (scipy mid-ranks + std-guard).

The Gen-2 rankers (MS3, Sobol, Task) were ported from
``scripts/sim2rank/meta_eval.py``, whose SROCC term is
``meta_eval._spearman_per_metric``: ``scipy.stats.spearmanr`` (TIE-CORRECTED
mid-ranks), guarded by ``np.all(isnan(v)) or np.std(nan_to_num(v)) < 1e-12 -> 0``
and ``|rho|``.

The in-core :func:`rankers.sim2rank._spearman_rho` is the *ordinal* (tie-blind,
``argsort(argsort)``) variant the Gen-1/Gen-3 rankers use, and it is NOT a
faithful drop-in for the legacy Gen-2 SROCC: on a **constant/saturated**
trajectory it returns ``|rho|=1`` instead of ``0`` (no std-guard), and on a
**tied** trajectory it diverges from scipy mid-ranks by up to ~0.1. Those cases
do occur on real data (a metric that saturates or flatlines over a sub-range),
so the Gen-2 ports must use this legacy-faithful helper — keeping the
Gen-1/Gen-3 ``_spearman_rho`` behaviour untouched. (Adversarial verification,
2026-05-30: the ordinal reuse made Sobol's headline score diverge by up to 0.69
on a saturated family; this restores byte-faithfulness to the legacy.)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np


def legacy_srocc(trajectory: Sequence[float] | np.ndarray, severity_grid) -> float:
    """``|Spearman(trajectory, severity)|`` faithful to ``_spearman_per_metric``.

    Mirrors the legacy contract exactly: all-NaN or near-constant input
    (population std ``< 1e-12``) → ``0.0``; otherwise scipy ``spearmanr`` with
    ``nan_policy='omit'`` under RuntimeWarning suppression, a NaN/None correlation
    or any exception → ``0.0``, then ``abs``.
    """
    import torch

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    if isinstance(severity_grid, torch.Tensor):
        severity_grid = severity_grid.detach().cpu().numpy()
    v = np.asarray(trajectory, dtype=np.float64).ravel()
    s = np.asarray(severity_grid, dtype=np.float64).ravel()
    if np.all(np.isnan(v)) or float(np.std(np.nan_to_num(v))) < 1e-12:
        return 0.0
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rho, _ = spearmanr(v, s, nan_policy="omit")
        except Exception:
            return 0.0
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return 0.0
    return float(abs(rho))


def srocc_signed(trajectory: Sequence[float] | np.ndarray, severity_grid) -> float:
    """Signed tie-corrected Spearman (scipy mid-ranks); 0.0 on degenerate input.

    Identical to :func:`legacy_srocc` but **preserves the sign** — LGDR needs the
    signed time-averaged correlation, not ``|rho|``, and the tie-blind
    ``argsort(argsort)`` it previously used returned ``|rho| = 1`` on saturated /
    tied trajectories (critique H2). All-NaN or near-constant input → ``0.0``.
    """
    import torch

    if isinstance(trajectory, torch.Tensor):
        trajectory = trajectory.detach().cpu().numpy()
    if isinstance(severity_grid, torch.Tensor):
        severity_grid = severity_grid.detach().cpu().numpy()
    v = np.asarray(trajectory, dtype=np.float64).ravel()
    s = np.asarray(severity_grid, dtype=np.float64).ravel()
    if np.all(np.isnan(v)) or float(np.std(np.nan_to_num(v))) < 1e-12:
        return 0.0
    if np.all(np.isnan(s)) or float(np.std(np.nan_to_num(s))) < 1e-12:
        return 0.0
    from scipy.stats import spearmanr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            rho, _ = spearmanr(v, s, nan_policy="omit")
        except Exception:
            return 0.0
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return 0.0
    return float(rho)
