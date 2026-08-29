"""Gen-2 information ranker — MI (KSG mutual information).

Exposes, as a standalone registered :class:`BaseRanker`, the KSG mutual
information axis :math:`I(\\text{metric}; \\text{severity})` from
``scripts/sim2rank/meta_eval.py`` (``compute_mi_score`` + the ``_ksg_mi`` kernel).
The Kraskov-Stögbauer-Grassberger algorithm-1 estimator measures how much
information a metric's trajectory carries about the underlying degradation
*severity* — a metric that tracks severity informatively (regardless of
monotonicity or linearity) scores high.

.. warning:: **Inputs are standardised before the k-NN search (issue #250).**

   KSG is a *metric* estimator: it counts neighbours inside a Chebyshev ball, so
   feeding it raw values makes the estimate a function of the metric's **units**.
   Verified on identical rank information: the same variable rescaled by ``c``
   returned MI ``0.0102 / 0.4219 / 0.0019`` for ``c = 1e-3 / 1 / 1e3``, where
   scikit-learn's reference KSG returns ``0.3999`` for all three. On a realistic
   pool that penalises an MSE-like metric (values ~1e-3) roughly 100x against an
   SSIM-like metric (values ~1) carrying exactly the same information.

   The default transform is the **copula** (rank → uniform marginals), which is
   invariant under *any* strictly monotone reparameterisation — the same
   invariance sim2rank claims for its rank-based statistics elsewhere. ``zscore``
   is the affine-only alternative and reproduces scikit-learn exactly; ``none``
   reproduces the bug and exists only so a regression test can exhibit it.

.. note:: **There is no conditional MI here.** The module used to advertise
   :math:`I(M; \\Theta \\mid C)` through a ``content_labels`` argument with **zero**
   call sites, while every production path computed the unconditional
   :math:`I(M; \\Theta)`. The dead branch also estimated the CMI as a *difference of
   two KSG estimates*, whose biases do not cancel. The claim is withdrawn rather
   than faked; a real CMI needs a Frenzel-Pompe partial-MI estimator.

Reuses the validated in-core trajectory helpers (``_build_trajectory_table`` /
``_per_family_average``) so the ``(metric, content, family, severity)`` table is
assembled identically to the Gen-1/Gen-3 rankers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..ranking_guards import assert_not_degenerate, order_from_scores
from ..types import MetricEvaluationDataset, MetricSet, RankingResult
from .base import BaseRanker
from .registry import register_ranker
from .sim2rank import _build_trajectory_table

#: Per-variable transforms applied before the k-NN search. Registered set — an
#: unknown value RAISES (no silent fallback to "none", pitfall #9).
KSG_TRANSFORMS = ("copula", "zscore", "none")


@dataclass
class MIConfig:
    k: int = 3  # KSG nearest-neighbour parameter (typical 3-5)
    transform: str = "copula"  # one of KSG_TRANSFORMS — see the module docstring

    def __post_init__(self) -> None:
        if self.transform not in KSG_TRANSFORMS:
            raise ValueError(
                f"Unknown KSG transform {self.transform!r}; expected one of {KSG_TRANSFORMS}."
            )


def _empty(method: str) -> RankingResult:
    return RankingResult(method=method, scores={}, ranks=[], diagnostics={})


# ---------------------------------------------------------------------------
# KSG kernel — kept in lock-step with scripts/sim2rank/meta_eval.py::_ksg_mi.
# The tie-break RNG seed is load-bearing. The marginal-ball radius uses a
# SCALE-INVARIANT relative open-ball tolerance ``eps*(1 - 1e-9)`` (Kraskov-1's
# strict ``< eps_i`` open ball): an absolute ``eps - 1e-12`` shrink is a large
# fraction of eps when eps is tiny (concentrated data), under-counting marginal
# neighbours and biasing MI down (critique M3). Any change here MUST be mirrored
# in the legacy ``_ksg_mi`` to preserve the parity test.
# ---------------------------------------------------------------------------


def _standardise_for_ksg(a: np.ndarray, transform: str) -> np.ndarray:
    """Put every column of ``a`` on a common scale before the Chebyshev k-NN search.

    Mirrors ``meta_eval._standardise_for_ksg``. See the module docstring for why
    this is not optional (issue #250).

    Raises:
        ValueError: On a transform outside :data:`KSG_TRANSFORMS`.
    """
    if transform not in KSG_TRANSFORMS:
        raise ValueError(f"Unknown KSG transform {transform!r}; expected one of {KSG_TRANSFORMS}.")
    if transform == "none":
        return a
    if transform == "zscore":
        sd = a.std(axis=0, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (a - a.mean(axis=0, keepdims=True)) / sd

    from scipy.stats import rankdata

    n = a.shape[0]
    return np.stack(
        [rankdata(a[:, j], method="average") / (n + 1.0) for j in range(a.shape[1])],
        axis=1,
    )


def _ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 3, transform: str = "copula") -> float:
    """Kraskov-Stögbauer-Grassberger algorithm 1 estimator of I(X; Y).

    Args:
        x: (N, d_x) sample matrix.
        y: (N, d_y) sample matrix.
        k: Nearest-neighbour parameter (typical 3-5).
        transform: Per-variable standardisation applied before the k-NN search —
            see :func:`_standardise_for_ksg`. Without it the estimator is not
            scale-invariant and MI ranks metrics by their units (issue #250).
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.atleast_2d(x.astype(np.float64))
    y = np.atleast_2d(y.astype(np.float64))
    if x.shape[1] > x.shape[0]:
        x = x.T
    if y.shape[1] > y.shape[0]:
        y = y.T
    N = x.shape[0]
    if y.shape[0] != N:
        raise ValueError("x and y must have the same number of rows")
    if k + 1 >= N:
        return 0.0

    x = _standardise_for_ksg(x, transform)
    y = _standardise_for_ksg(y, transform)

    # Tiny noise breaks ties (KSG requires distinct distances). Applied AFTER the
    # transform, so its magnitude is fixed relative to the standardised scale.
    rng = np.random.default_rng(0)
    x = x + 1e-10 * rng.standard_normal(x.shape)
    y = y + 1e-10 * rng.standard_normal(y.shape)
    z = np.hstack([x, y])

    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    eps, _ = tree_z.query(z, k=k + 1, p=np.inf)
    eps = eps[:, -1]  # k-th NN distance in joint space (Chebyshev)

    # Scale-invariant open-ball radius (strictly inside the k-th-NN shell).
    r_ball = eps * (1.0 - 1e-9)
    nx = np.array([len(tree_x.query_ball_point(x[i], r=r_ball[i], p=np.inf)) for i in range(N)])
    ny = np.array([len(tree_y.query_ball_point(y[i], r=r_ball[i], p=np.inf)) for i in range(N)])
    # Subtract 1 because the point itself is counted
    nx = np.maximum(nx - 1, 0)
    ny = np.maximum(ny - 1, 0)

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)
    return max(0.0, float(mi))


# ---------------------------------------------------------------------------
# Per-trajectory wrapper — mirrors compute_mi_score's per-row body.
# NaN-fill, constant guard, reshape to (T, 1).
# ---------------------------------------------------------------------------


def _mi_for_trajectory(
    trajectory: np.ndarray, severity: np.ndarray, k: int, transform: str = "copula"
) -> float:
    """KSG MI between a metric sample vector and the matching severity vector.

    Mirrors the per-row body of the legacy ``compute_mi_score``: ``nan_to_num`` the
    values, return 0 for a constant vector (std < 1e-12), otherwise reshape to
    ``(N, 1)`` columns and call :func:`_ksg_mi`. ``trajectory`` and ``severity``
    must be the same length — they are *paired samples*, not a curve and a grid.
    """
    v = np.nan_to_num(trajectory).reshape(-1, 1)
    if v.std() < 1e-12:
        return 0.0
    s = np.asarray(severity, dtype=np.float64).reshape(-1, 1)
    return _ksg_mi(v, s, k=k, transform=transform)


def _pooled_family_samples(
    traj: dict[tuple[str, str], list[float]],
    family: str,
    contents: list[str],
    severities: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a family's ``(content, severity)`` cells into paired sample vectors.

    Returns ``(values, severity_of_each_value)`` over all contents, dropping cells
    the simulator never filled (NaN).

    **Why pool instead of averaging (issue #250, found while wiring the degeneracy
    guard).** The ranker used to estimate MI from the *content-averaged*
    trajectory, i.e. from ``T`` points. KSG returns ``0.0`` outright when
    ``k + 1 >= N``, so on this pipeline's standard 4-severity datasets (``T = 4``,
    ``k = 3``) the estimate was **identically zero for every metric** — an inert
    axis that still printed a leaderboard, and exactly the kind of silent facade
    :func:`assert_not_degenerate` exists to catch. Averaging over contents also
    destroys the very variability MI is meant to weigh: a metric whose value
    depends more on *which subject* it sees than on the severity carries little
    information about severity, and the average hides that. Pooling gives
    ``N = C·T`` genuine ``(metric, severity)`` realisations, which is what the
    estimator's own docstring always claimed to consume.
    """
    values: list[float] = []
    sev: list[float] = []
    for c in contents:
        row = traj.get((c, family))
        if row is None:
            continue
        for t, v in enumerate(row):
            if np.isfinite(v):
                values.append(float(v))
                sev.append(float(severities[t]))
    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(sev, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


@register_ranker(
    "mi",
    generation=2,
    description="KSG mutual information I(metric; severity), rank/copula-transformed",
)
class MIRanker(BaseRanker):
    method_name = "MI"

    def __init__(self, config: MIConfig | None = None) -> None:
        self.config = config or MIConfig()

    def rank(self, metric_set: MetricSet, dataset: MetricEvaluationDataset) -> RankingResult:
        if dataset.n_samples == 0:
            return _empty(self.method_name)

        k = self.config.k
        transform = self.config.transform
        scores: dict[str, float] = {}
        per_family_mi: dict[str, dict[str, float]] = {}

        for name in metric_set.names():
            traj, severities, contents, families = _build_trajectory_table(dataset, name)

            breakdown: dict[str, float] = {}
            for fam in families:
                vals, sev = _pooled_family_samples(traj, fam, contents, severities)
                if vals.size == 0:
                    continue
                breakdown[fam] = _mi_for_trajectory(vals, sev, k, transform=transform)
            per_family_mi[name] = breakdown
            scores[name] = (
                float(sum(breakdown.values()) / max(len(breakdown), 1)) if breakdown else 0.0
            )

        # A KSG estimate on a T-point trajectory needs T > k + 1 samples; below that
        # the kernel returns 0.0 for EVERY metric and the "ranking" is pool order.
        # The guard turns that into a loud error instead of a fabricated leaderboard.
        assert_not_degenerate(list(scores.values()), name=self.method_name)
        return RankingResult(
            method=self.method_name,
            scores=scores,
            ranks=order_from_scores(scores),
            diagnostics={
                "per_family_mi": per_family_mi,
                "k": k,
                "transform": transform,
            },
        )


__all__ = ["KSG_TRANSFORMS", "MIConfig", "MIRanker", "_ksg_mi"]
