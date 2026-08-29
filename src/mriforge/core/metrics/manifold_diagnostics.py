"""Estimator-dependent manifold diagnostics for k-space cold diffusion.

The cold-diffusion papers' design corollaries C2/C3 and the certified error
bound are all stated in terms of quantities of the admissible sets
``S_{t-1}`` — the reach ``tau``, the step-to-reach ratio
``kappa_t = delta_t / tau_{t-1}``, the tangential defect ``theta_t``, and the
manifold departure ``mu``.  None of these is exactly computable from data;
every function here is a plug-in *estimator*, and its output is a diagnostic,
never a certificate (the exactly-computable trust layer lives in
``trust_metrics.py`` / ``trajectory_metrics.py``).

Design:
    - ``estimate_reach()`` — point-cloud reach via the criterion
      ``<v, y-p> <= ||y-p||^2 / (2 tau)``: ``tau_hat`` is the min over ordered
      pairs of ``||y-p||^2 / (2 ||(y-p)_normal||)`` with local-PCA tangents.
    - ``step_budget_ratio()`` — C2 verdict object for one level:
      ``kappa = delta / tau_hat``, well-posedness (``kappa < 1``), the C2 cap
      (``kappa <= 1/2``, so the Lipschitz factor ``Lambda <= 2``), and
      ``Lambda = 1 / (1 - kappa)``.
    - ``tangential_defect()`` — ``theta_hat``: supremum over samples of the
      tangential share ``||P_T w|| / ||w||`` of the displacement ``w``.
    - ``manifold_departure()`` — ``mu_hat``: distance from a reconstruction to
      the nearest member of a reference set standing in for the manifold.
    - ``certified_error_bound()`` — ``mu + omega * (kappa + L * mu)``, the
      linear-modulus form of the papers' bound
      ``d(x_hat, x_0) <= mu + omega_t(kappa_t + L_t mu)``.

All functions are stateless and operate on NumPy arrays, matching
``statistical_tests.py`` (offline cohort surface, not the training loop).

References:
    - Federer, H. (1959). Curvature Measures. (reach)
    - Aamari, E. et al. (2019). Estimating the reach of a manifold.
    - Moreau, J.-J. (1962). Décomposition orthogonale d'un espace hilbertien
      selon deux cônes mutuellement polaires. (tangential/normal split)
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "certified_error_bound",
    "estimate_reach",
    "manifold_departure",
    "step_budget_ratio",
    "tangential_defect",
]

# Pairs whose normal component is this small (relative to the chord) are
# tangentially aligned: their ratio estimates nothing about curvature and
# would inject numerical noise into the min.
_NORMAL_FLOOR_REL = 1e-9


def _as_2d(points: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array (n_points, ambient_dim)")
    return arr


def _tangent_bases(points: np.ndarray, intrinsic_dim: int, k_neighbors: int) -> np.ndarray:
    """Local-PCA tangent basis at every point: (n, ambient, intrinsic)."""
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    bases = np.empty((n, points.shape[1], intrinsic_dim))
    for i in range(n):
        # k nearest neighbours *excluding* the point itself (index 0 of argsort).
        neighbours = points[np.argsort(d2[i])[1 : k_neighbors + 1]]
        centred = neighbours - neighbours.mean(axis=0)
        # Rows of Vt are principal directions; the top intrinsic_dim span T_p.
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        bases[i] = vt[:intrinsic_dim].T
    return bases


def estimate_reach(
    points: np.ndarray, intrinsic_dim: int = 1, k_neighbors: int | None = None
) -> float:
    """Plug-in reach estimate of the manifold sampled by ``points``.

    The reach criterion says ``<v, y-p> <= ||y-p||^2 / (2 tau)`` for every
    pair ``p, y`` on the set and unit normal ``v`` at ``p``; inverting it over
    all ordered pairs gives ``tau_hat = min ||y-p||^2 / (2 ||(y-p)_normal||)``
    with the normal component measured against a local-PCA tangent basis.

    Returns ``inf`` for an affine cloud (no pair has a normal component —
    a flat set has infinite reach), never raises for one.
    """
    pts = _as_2d(points, "points")
    n, ambient = pts.shape
    if not 1 <= intrinsic_dim < ambient:
        raise ValueError("intrinsic_dim must satisfy 1 <= intrinsic_dim < ambient_dim")
    if k_neighbors is None:
        k_neighbors = min(n - 1, max(intrinsic_dim + 2, 10))
    if k_neighbors <= intrinsic_dim:
        raise ValueError("k_neighbors must exceed intrinsic_dim")
    if n < k_neighbors + 1:
        raise ValueError("need at least k_neighbors + 1 points")

    bases = _tangent_bases(pts, intrinsic_dim, k_neighbors)
    tau_hat = math.inf
    for i in range(n):
        chords = np.delete(pts, i, axis=0) - pts[i]  # (n-1, ambient)
        tangential = chords @ bases[i] @ bases[i].T
        normal_norms = np.linalg.norm(chords - tangential, axis=1)
        chord_sq = np.sum(chords**2, axis=1)
        valid = normal_norms > _NORMAL_FLOOR_REL * np.sqrt(chord_sq)
        if np.any(valid):
            tau_hat = min(tau_hat, float(np.min(chord_sq[valid] / (2.0 * normal_norms[valid]))))
    return tau_hat


def step_budget_ratio(delta_t: float, tau_hat: float) -> dict[str, float | bool]:
    """C2 verdict for one level: ``kappa_t = delta_t / tau_hat`` and friends.

    ``well_posed`` is the step condition ``kappa < 1`` (single-valued reverse
    projection); ``satisfies_c2`` is the design cap ``kappa <= 1/2`` that
    bounds the per-step Lipschitz factor ``amplification = 1/(1-kappa)`` by 2.
    """
    if not tau_hat > 0.0:
        raise ValueError("tau_hat must be > 0")
    if delta_t < 0.0:
        raise ValueError("delta_t must be >= 0")
    kappa = delta_t / tau_hat
    return {
        "kappa": kappa,
        "well_posed": kappa < 1.0,
        "satisfies_c2": kappa <= 0.5,
        "amplification": 1.0 / (1.0 - kappa) if kappa < 1.0 else math.inf,
    }


def tangential_defect(
    base_points: np.ndarray,
    displacements: np.ndarray,
    intrinsic_dim: int = 1,
    k_neighbors: int | None = None,
) -> float:
    """``theta_hat``: supremal tangential share of the degradation step.

    ``base_points`` sample ``S_{t-1}`` and ``displacements[i]`` is
    ``w(y_i) = D_{t-1->t}(y_i) - y_i``.  Each ``w`` is split against the
    local-PCA tangent basis at its base point; the paper's ``theta_t`` is a
    supremum, so the estimate is the max (not the mean) of
    ``||P_T w|| / ||w||`` over samples with ``w != 0``.  Returns 0.0 when
    every displacement is zero (the identity step has no defect).
    """
    pts = _as_2d(base_points, "base_points")
    disp = _as_2d(displacements, "displacements")
    if disp.shape != pts.shape:
        raise ValueError("displacements must align with base_points")
    n, ambient = pts.shape
    if not 1 <= intrinsic_dim < ambient:
        raise ValueError("intrinsic_dim must satisfy 1 <= intrinsic_dim < ambient_dim")
    if k_neighbors is None:
        k_neighbors = min(n - 1, max(intrinsic_dim + 2, 10))
    if k_neighbors <= intrinsic_dim:
        raise ValueError("k_neighbors must exceed intrinsic_dim")
    if n < k_neighbors + 1:
        raise ValueError("need at least k_neighbors + 1 points")

    bases = _tangent_bases(pts, intrinsic_dim, k_neighbors)
    norms = np.linalg.norm(disp, axis=1)
    theta_hat = 0.0
    for i in np.flatnonzero(norms > 0.0):
        tangential = bases[i] @ (bases[i].T @ disp[i])
        theta_hat = max(theta_hat, float(np.linalg.norm(tangential) / norms[i]))
    return theta_hat


def manifold_departure(x_hat: np.ndarray, reference_set: np.ndarray) -> float:
    """``mu_hat = d(x_hat, M)``: distance to the nearest reference sample.

    The reference set is the working model of the manifold ``M``; the papers
    are explicit that the bound below is only as good as this model.
    """
    refs = _as_2d(reference_set, "reference_set")
    x = np.asarray(x_hat, dtype=np.float64).reshape(-1)
    if x.shape[0] != refs.shape[1]:
        raise ValueError("x_hat must have the reference set's ambient dimension")
    if refs.shape[0] == 0:
        raise ValueError("reference_set must contain at least one sample")
    return float(np.min(np.linalg.norm(refs - x, axis=1)))


def certified_error_bound(mu_hat: float, kappa_T: float, omega_T: float, L_T: float = 1.0) -> float:
    """``mu + omega_T (kappa_T + L_T mu)``: the linear-modulus error bound.

    Converts two *measurable* quantities — the departure ``mu_hat`` and the
    endpoint inconsistency ``kappa_T`` — into a bound on the unmeasurable
    true error ``d(x_hat, x_0)``, at the price of the structural constants
    ``omega_T`` (modulus of resolution, linear form) and ``L_T`` (degradation
    Lipschitz constant).  Monotone in every argument.
    """
    for name, value in (
        ("mu_hat", mu_hat),
        ("kappa_T", kappa_T),
        ("omega_T", omega_T),
        ("L_T", L_T),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")
    return mu_hat + omega_T * (kappa_T + L_T * mu_hat)
