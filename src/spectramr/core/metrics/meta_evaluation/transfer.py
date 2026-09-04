"""Sim-to-real ranking-transfer certificate (Layer L4).

L1–L3 certify the metric ranking against the *simulated* law; this module is the
*certificate of external validity* that the simulated ranking still holds under the
*real* acquisition law. It is the executable shadow of the Lean module
``Sim2Rank.Transfer`` (``docs/lean/Sim2Rank/Sim2Rank/Transfer.lean``):

* :func:`tv_distance` is the discrete total variation of Lean ``tv``.
* :func:`pmean_diff_bound` is the bounded-functional transfer bound
  ``|τ(P) − τ(Q)| ≤ 4·C·TV(P, Q)`` of Lean ``pmean_diff_le_tv``.
* :func:`tv_transfer_certificate` is the headline ``ranking_transfer``: the
  simulated ranking of a pair transfers to the real law when
  ``TV(P_sim, P_real) < Δ/8`` (``Δ`` = the simulated population gap).

The *joint*-law TV the theorem needs is bounded below by two estimable quantities,
and the pipeline takes their max (the tightest available lower bound):

* the 1-D **marginal** score-law TV (:func:`tv_distance_samples`) — the TV of the
  marginal the rank statistic consumes. A marginal TV never exceeds the joint TV, so
  on its own it is *optimistic* (the real law could differ more than any marginal
  shows). Its *plug-in histogram* estimator is also **biased upward** at finite
  sample size (two samples from the *same* law give ``E[TV] > 0``,
  ``O(sqrt(n_bins/n))``), so :func:`tv_distance_samples` supports adaptive
  (``n``-aware) binning and an opt-in pooled-null debias so a perfect twin tends to
  ``TV -> 0`` (critique A5).
* the **joint**-law TV (:func:`tv_distance_joint_samples`) — a 1-NN classifier
  two-sample test on the multivariate metric vectors, a tighter lower bound that
  catches a sim->real shift hiding in the *dependence structure* even when every
  marginal matches (the joint-vs-marginal gap).

NumPy-free, torch-free: pure ``math``/lists (+ stdlib ``random`` for the optional
debias bootstrap) so it imports anywhere, matching :mod:`powered`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "TransferCertificate",
    "pmean_diff_bound",
    "tv_distance",
    "tv_distance_joint_samples",
    "tv_distance_samples",
    "tv_transfer_certificate",
]


def tv_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Discrete total-variation distance ``½·∑ |p_i − q_i|`` (Lean ``tv``).

    ``p`` and ``q`` are probability vectors of equal length (each ``≥ 0``,
    summing to ``1``).
    """
    if len(p) != len(q):
        raise ValueError("p and q must have equal length")
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q, strict=True))


def tv_distance_samples(
    sim_values: Sequence[float],
    real_values: Sequence[float],
    *,
    n_bins: int | None = 32,
    debias: bool = False,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """Estimable 1-D score-law TV between two empirical samples.

    Bins both samples on a shared range and returns the TV of the normalized
    histograms — the marginal score-law TV the rank statistic consumes (a lower
    bound on the joint-law TV of Lean ``ranking_transfer``). Disjoint supports give
    TV ``= 1``; identical samples give ``≈ 0``.

    ``n_bins``: fixed bin count, or ``None`` for an **adaptive** ``~sqrt(min sample
    size)`` rule (bounded to ``[4, 64]``). Adaptive binning keeps the plug-in
    histogram-TV bias ``O(sqrt(n_bins/n))`` shrinking as ``n`` grows.

    ``debias``: when True, subtract the **pooled-null** expected histogram-TV — the
    average TV between two bootstrap resamples drawn from the *pooled* sample,
    matched to the actual sample sizes. This corrects the upward finite-sample bias
    so two draws from the *same* law tend to ``TV -> 0`` (critique A5). Deterministic
    (seeded). Off by default so the population-TV edge cases stay exact.

    Non-finite values (``NaN`` / ``±inf``) are dropped before binning. This honors
    the *crash→NaN contract* of :func:`simulator.precompute_metric_values` (and the
    real-cohort builder ``_build_real_metric_values``): a metric that crashes on a
    sample is recorded as ``NaN``, and every downstream consumer must mask non-finite
    values — ``int(NaN)`` raises ``ValueError`` and a raw ``min``/``max`` over a list
    containing ``NaN`` returns an order-dependent (possibly ``NaN``) bound that
    poisons the bin width. Mirrors the masking every ranker already does.
    """
    if not sim_values or not real_values:
        raise ValueError("both samples must be non-empty")
    sim_finite = [v for v in sim_values if math.isfinite(v)]
    real_finite = [v for v in real_values if math.isfinite(v)]
    if not sim_finite or not real_finite:
        raise ValueError(
            "both samples must contain at least one finite value (all were "
            "NaN/inf — the metric likely crashed on the whole cohort)"
        )
    n_sim, n_real = len(sim_finite), len(real_finite)
    if n_bins is None:
        n_bins = max(4, min(64, round(min(n_sim, n_real) ** 0.5)))
    lo = min(min(sim_finite), min(real_finite))
    hi = max(max(sim_finite), max(real_finite))
    if hi <= lo:
        return 0.0  # both samples are a single shared point
    width = (hi - lo) / n_bins
    bins = n_bins

    def _hist(vals: Sequence[float]) -> list[float]:
        counts = [0.0] * bins
        for v in vals:
            k = min(int((v - lo) / width), bins - 1)
            counts[k] += 1.0
        total = sum(counts)
        return [c / total for c in counts]

    cross_tv = tv_distance(_hist(sim_finite), _hist(real_finite))
    if not debias:
        return cross_tv

    # Pooled-null bias: E[TV] when both samples come from the SAME (pooled) law,
    # matched to (n_sim, n_real). Subtract it so a perfect twin tends to 0.
    import random

    rng = random.Random(seed)
    pooled = sim_finite + real_finite
    m = len(pooled)
    acc = 0.0
    for _ in range(n_boot):
        a = [pooled[rng.randrange(m)] for _ in range(n_sim)]
        b = [pooled[rng.randrange(m)] for _ in range(n_real)]
        acc += tv_distance(_hist(a), _hist(b))
    bias = acc / n_boot
    return max(0.0, cross_tv - bias)


def tv_distance_joint_samples(
    sim_matrix: Sequence[Sequence[float]],
    real_matrix: Sequence[Sequence[float]],
    *,
    debias: bool = True,
    n_perm: int = 50,
    seed: int = 0,
) -> float:
    """Multivariate (joint-law) TV lower bound via a 1-NN classifier two-sample test.

    The per-metric marginal TV (:func:`tv_distance_samples`) is a LOWER bound on the
    *joint*-law TV the transfer theorem needs — a marginal TV never exceeds the joint
    TV — so a certificate built on marginals is optimistic. This estimates the joint
    TV directly on the multivariate metric-value vectors: by the Le Cam / classifier
    two-sample-test identity, ``TV(P, Q) >= 2*A* - 1`` where ``A*`` is the best
    classifier's balanced accuracy distinguishing P from Q. We estimate ``A*`` with a
    leave-one-out 1-nearest-neighbour classifier (Lopez-Paz & Oquab, 2017) on the
    pooled, per-dimension-standardised samples. This is a **tighter** (larger) lower
    bound on the joint TV than the marginal max, so the transfer certificate is less
    optimistic.

    ``sim_matrix`` / ``real_matrix`` are ``[N, M]`` (N samples x M metrics). Columns
    that are entirely non-finite are dropped; remaining non-finite entries are
    column-median-imputed (crash->NaN contract). With ``debias`` (default), the
    label-permuted null balanced accuracy is subtracted so two draws from the *same*
    law give ``TV -> 0``. Deterministic (seeded). Returns 0 on degenerate input.
    """
    import numpy as np

    sim = np.asarray(sim_matrix, dtype=np.float64)
    real = np.asarray(real_matrix, dtype=np.float64)
    if sim.size == 0 or real.size == 0:
        return 0.0  # nothing to compare
    if sim.ndim != 2 or real.ndim != 2:
        raise ValueError("sim_matrix and real_matrix must be 2-D [N, M]")
    if sim.shape[1] != real.shape[1]:
        raise ValueError("sim and real must have the same number of metrics (M)")
    n_sim, n_real = sim.shape[0], real.shape[0]
    if n_sim < 2 or n_real < 2 or sim.shape[1] == 0:
        return 0.0

    pooled = np.vstack([sim, real])
    # Drop all-NaN columns; median-impute the rest; standardise per dimension.
    finite_col = np.isfinite(pooled).any(axis=0)
    pooled = pooled[:, finite_col]
    if pooled.shape[1] == 0:
        return 0.0
    med = np.nanmedian(np.where(np.isfinite(pooled), pooled, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    pooled = np.where(np.isfinite(pooled), pooled, med)
    std = pooled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    pooled = (pooled - pooled.mean(axis=0)) / std

    labels = np.concatenate([np.zeros(n_sim), np.ones(n_real)])

    def _loo_balanced_accuracy(lab: np.ndarray) -> float:
        from scipy.spatial import cKDTree

        tree = cKDTree(pooled)
        # Two nearest neighbours: index 0 is the point itself; index 1 is its LOO NN.
        _, idx = tree.query(pooled, k=2, p=2)
        pred = lab[idx[:, 1]]
        # Balanced accuracy: mean of per-class recall (robust to class imbalance).
        recalls = []
        for cls in (0.0, 1.0):
            mask = lab == cls
            if not mask.any():
                return 0.5
            recalls.append(float((pred[mask] == cls).mean()))
        return float(np.mean(recalls))

    acc = _loo_balanced_accuracy(labels)
    if not debias:
        return max(0.0, 2.0 * acc - 1.0)

    rng = np.random.default_rng(seed)
    null_accs = [_loo_balanced_accuracy(rng.permutation(labels)) for _ in range(n_perm)]
    null_acc = float(np.mean(null_accs))
    # Debiased TV = (2*acc-1) - (2*null_acc-1) = 2*(acc - null_acc), floored at 0.
    return max(0.0, 2.0 * (acc - null_acc))


def pmean_diff_bound(tv: float, *, kernel_bound: float = 1.0) -> float:
    """Bounded-functional transfer bound ``4·C·TV`` (Lean ``pmean_diff_le_tv``).

    The most the population value of a kernel bounded by ``C = kernel_bound`` can
    move between two laws at total-variation distance ``tv``.
    """
    if tv < 0:
        raise ValueError(f"tv must be >= 0, got {tv}")
    return 4.0 * kernel_bound * tv


@dataclass
class TransferCertificate:
    """Per-pair sim-to-real transfer badge."""

    better: str
    worse: str
    sim_gap: float
    tv: float
    budget: float  # = sim_gap / 8
    transfers: bool

    def __post_init__(self) -> None:
        # The certificate fires iff the twin is inside the transfer budget.
        object.__setattr__(self, "transfers", self.tv < self.budget)


def tv_transfer_certificate(sim_scores: dict[str, float], tv: float) -> list[TransferCertificate]:
    """Per-pair sim-to-real ranking-transfer certificates (Lean ``ranking_transfer``).

    For every ordered pair ``(a, b)`` with simulated gap ``Δ = |s_a − s_b|``, the
    simulated ordering transfers to the real law iff ``tv < Δ/8``, where ``tv`` is
    the (estimated) total-variation distance between the simulated and real laws.
    At ``tv = 0`` (perfect digital twin) every strictly-ordered pair transfers,
    recovering the sim-only guarantee (Lean ``ranking_transfer_recovers_sim``).
    """
    if tv < 0:
        raise ValueError(f"tv must be >= 0, got {tv}")
    names = list(sim_scores)
    out: list[TransferCertificate] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            gap = sim_scores[a] - sim_scores[b]
            better, worse = (a, b) if gap >= 0 else (b, a)
            delta = abs(gap)
            out.append(
                TransferCertificate(
                    better=better,
                    worse=worse,
                    sim_gap=delta,
                    tv=tv,
                    budget=delta / 8.0,
                    transfers=False,  # set by __post_init__
                )
            )
    return out
