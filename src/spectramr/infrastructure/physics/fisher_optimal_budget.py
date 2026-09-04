r"""Fisher-optimal cross-dimensional acquisition budget (A-2.3 *FOBA*).

When an acquisition has a fixed total budget (number of measurements / time) to
split across several **dimensions** — spatial averages, temporal frames, extra
contrasts, k-space lines — how should it be allocated to learn the parameters of
interest most precisely? FOBA answers this as a **D-optimal experiment design**:
allocate the budget to maximise the log-determinant of the (diagonal) Fisher
information matrix, which minimises the volume of the parameter Cramér-Rao
confidence ellipsoid.

The framework already had the *Fisher information* machinery (Bloch Jacobians,
``fisher_rao_geometry``, ``marker_estimators``) and a single-dimension CRLB pulse
designer, but no **cross-dimensional budget allocator**. This module is that
missing convex solver.

**Model.** Dimension :math:`d` arrives with a baseline Fisher information
:math:`a_d \ge 0` (from measurements already planned) and gains :math:`r_d > 0`
per unit of new budget (the Fisher information of one measurement in that
dimension — for i.i.d. averaging, FI is additive). The D-optimal objective is

.. math::

    \max_{b\ge 0}\ \sum_d \log\!\big(a_d + r_d\,b_d\big)
    \quad\text{s.t.}\quad \sum_d b_d = B .

This is concave, so the KKT conditions give a **water-filling** solution: at the
optimum the marginal gains :math:`r_d/(a_d+r_d b_d)` are equal to a single water
level :math:`\lambda` across all *active* dimensions, i.e.
:math:`b_d=\max\!\big(0,\ 1/\lambda - a_d/r_d\big)`. The scalar :math:`\lambda` is
found by bisection so the budget balances. A dimension that already has high
baseline information (large :math:`a_d/r_d`) receives less — or zero — new budget.

**Connecting to the physics.** :func:`fisher_rate_from_jacobian` turns a Bloch /
forward-model Jacobian :math:`J=\partial(\text{signal})/\partial(\text{params})`
(e.g. from
:class:`~spectramr.models.blocks.differentiable_bloch.DifferentiableBlochLayer`)
into the per-parameter Fisher rate :math:`\mathrm{diag}(J^\top J)` (unit-variance
noise) — the :math:`r_d` of one measurement. FOBA is a *design-time* optimiser
(it sizes an acquisition), not a training paradigm, so it ships as a primitive +
this physics adapter, not a training strategy.
"""

from __future__ import annotations

import torch
from torch import Tensor


def fisher_rate_from_jacobian(jacobian: Tensor, noise_var: float = 1.0) -> Tensor:
    r"""Per-parameter Fisher information of one measurement: ``diag(J^T J) / sigma^2``.

    Args:
        jacobian: ``[n_obs, n_params]`` (real) signal Jacobian
            :math:`\partial s/\partial\theta`. Complex Jacobians are split into
            stacked real/imag rows (the FI of a complex Gaussian observation).
        noise_var: Measurement noise variance :math:`\sigma^2` (>0).

    Returns:
        ``[n_params]`` non-negative Fisher rates.
    """
    if noise_var <= 0:
        raise ValueError(f"noise_var must be > 0, got {noise_var}")
    j = jacobian
    if j.is_complex():
        j = torch.cat([j.real, j.imag], dim=0)
    if j.dim() != 2:
        raise ValueError(f"jacobian must be 2-D [n_obs, n_params], got shape {tuple(j.shape)}")
    return (j.pow(2).sum(dim=0) / float(noise_var)).clamp_min(0.0)


def d_optimal_budget(
    baseline_fisher: Tensor,
    rate_fisher: Tensor,
    budget: float,
    *,
    min_per_dim: float = 0.0,
    n_bisect: int = 80,
    eps: float = 1e-12,
) -> Tensor:
    r"""D-optimal water-filling budget allocation over dimensions.

    Maximises :math:`\sum_d \log(a_d + r_d b_d)` s.t. :math:`\sum_d b_d = B` and
    :math:`b_d \ge \text{min\_per\_dim}`, by bisection on the water level
    :math:`\lambda` (so :math:`b_d=\max(\text{min},\,1/\lambda - a_d/r_d)`).

    Args:
        baseline_fisher: ``[D]`` baseline Fisher info :math:`a_d \ge 0`.
        rate_fisher: ``[D]`` per-unit Fisher rate :math:`r_d > 0`.
        budget: total budget :math:`B > 0` to distribute.
        min_per_dim: floor on each dimension's allocation (must be feasible:
            ``D * min_per_dim <= budget``).
        n_bisect: bisection iterations for the water level.

    Returns:
        ``[D]`` allocation summing to ``budget``.
    """
    a = baseline_fisher.detach().float().reshape(-1)
    r = rate_fisher.detach().float().reshape(-1)
    d = a.numel()
    if r.numel() != d:
        raise ValueError(f"baseline_fisher and rate_fisher must match: {d} vs {r.numel()}")
    if budget <= 0:
        raise ValueError(f"budget must be > 0, got {budget}")
    if (r <= 0).any():
        raise ValueError(
            "all rate_fisher entries must be > 0 (a dimension with 0 rate gains nothing)"
        )
    if d * min_per_dim > budget + eps:
        raise ValueError(f"infeasible floor: D*min_per_dim={d * min_per_dim} > budget={budget}")

    # Budget above the floor is what water-filling distributes; the floor is added back.
    free_budget = budget - d * min_per_dim
    ratio = a / r  # the a_d/r_d offset per dim

    def allocation_for(lam: float) -> Tensor:
        # b_d = max(0, 1/lam - a_d/r_d) over the free budget (floor added later).
        return torch.clamp(1.0 / lam - ratio, min=0.0)

    # Bisect lambda so sum(allocation) == free_budget. Smaller lambda -> more budget.
    # Upper bound: lambda where every dim is just inactive (b=0): lam = 1/min(ratio)
    # gives all-zero; we need a lam small enough that sum >= free_budget.
    lam_hi = 1.0 / (float(ratio.min()) + eps) if free_budget > 0 else 1.0 / eps
    lam_lo = lam_hi
    # Shrink lam_lo until the allocation exceeds the free budget.
    for _ in range(200):
        if float(allocation_for(lam_lo).sum()) >= free_budget:
            break
        lam_lo *= 0.5
        if lam_lo < eps:
            break
    for _ in range(n_bisect):
        lam = 0.5 * (lam_lo + lam_hi)
        total = float(allocation_for(lam).sum())
        if total > free_budget:
            lam_lo = lam  # too much budget -> raise lambda
        else:
            lam_hi = lam
    b = allocation_for(0.5 * (lam_lo + lam_hi))
    # Rescale to hit the free budget exactly (bisection residual), then add the floor.
    s = float(b.sum())
    b = b * (free_budget / s) if s > eps else torch.full_like(a, free_budget / d)
    return b + min_per_dim


def d_optimal_logdet(baseline_fisher: Tensor, rate_fisher: Tensor, allocation: Tensor) -> float:
    r"""The D-optimal objective :math:`\sum_d \log(a_d + r_d b_d)` for an allocation."""
    a = baseline_fisher.detach().float().reshape(-1)
    r = rate_fisher.detach().float().reshape(-1)
    b = allocation.detach().float().reshape(-1)
    return float(torch.log(a + r * b).sum())


class FisherOptimalBudgetAllocator:
    """D-optimal cross-dimensional acquisition budget allocator (A-2.3 FOBA).

    Thin object wrapper over :func:`d_optimal_budget` that also reports the
    achieved log-det objective and the uniform-allocation baseline (so a caller
    can verify the Fisher-optimal allocation beats a naive equal split).
    """

    def __init__(self, min_per_dim: float = 0.0, n_bisect: int = 80) -> None:
        self.min_per_dim = float(min_per_dim)
        self.n_bisect = int(n_bisect)

    def allocate(
        self, baseline_fisher: Tensor, rate_fisher: Tensor, budget: float
    ) -> dict[str, object]:
        """Return ``{allocation, logdet, uniform_logdet, water_level_gains}``."""
        alloc = d_optimal_budget(
            baseline_fisher,
            rate_fisher,
            budget,
            min_per_dim=self.min_per_dim,
            n_bisect=self.n_bisect,
        )
        a = baseline_fisher.detach().float().reshape(-1)
        r = rate_fisher.detach().float().reshape(-1)
        uniform = torch.full_like(a, budget / a.numel())
        marginal = r / (a + r * alloc)  # equal across active dims at the optimum
        return {
            "allocation": alloc,
            "logdet": d_optimal_logdet(a, r, alloc),
            "uniform_logdet": d_optimal_logdet(a, r, uniform),
            "marginal_gains": marginal,
        }


__all__ = [
    "FisherOptimalBudgetAllocator",
    "d_optimal_budget",
    "d_optimal_logdet",
    "fisher_rate_from_jacobian",
]
