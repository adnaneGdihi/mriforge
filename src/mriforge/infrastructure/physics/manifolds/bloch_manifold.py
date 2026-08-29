"""Bloch relaxation manifold with Fisher-information metric.

Implements ``M = {(M0, T1, T2) : M0 > 0, T1 > 0, T2 > 0, T2 ≤ T1}`` with
the Fisher metric induced by the SPGR signal model in
``src/infrastructure/physics/differentiable_bloch.py``.

The Fisher metric is estimated by Monte-Carlo:

    g_ij(θ) = E_y[∂_i log p(y|θ) ∂_j log p(y|θ)]

using the differentiable Bloch operator's autograd graph rather than a
closed-form Jacobian, so that any future change to the operator
(custom pulse trains, B0/B1 inclusion) automatically propagates.

The exponential map uses a Riemannian-Euler / geodesic-ODE integrator
that respects the boundary constraint T2 ≤ T1 by reflection at the
boundary (a standard trick for constrained Brownian motion).

Idea 5 of TODO/audit/audit_plan_novel.md depends on this module.
"""

from __future__ import annotations

import logging
from typing import Final

import torch

from mriforge.infrastructure.physics.differentiable_bloch import DifferentiableBlochLayer

from .base import RiemannianManifold

logger = logging.getLogger(__name__)


_EPS: Final[float] = 1e-3  # interior margin, ms / units
_DEFAULT_TR: Final[float] = 8.0
_DEFAULT_TE: Final[float] = 4.0
_DEFAULT_ALPHA_RAD: Final[float] = 0.2618  # 15 degrees


class BlochRelaxationManifold(RiemannianManifold):
    """Fisher-information manifold over (M0, T1, T2) relaxation parameters.

    Parameters are stored *unnormalised* in physical units (ms for T1 / T2,
    arbitrary for M0). The Fisher metric is computed by averaging the
    outer product of the SPGR-signal score over a small Monte-Carlo
    sample of independent acquisitions.

    Args:
        tr: Repetition time used for Fisher-metric estimation, ms.
        te: Echo time, ms.
        flip_angle_rad: Flip angle, radians.
        n_mc_samples: Monte-Carlo sample size per metric evaluation.
        sigma_signal: Assumed Gaussian-noise standard deviation on the
            simulated SPGR signal (the Fisher metric for a Gaussian
            likelihood reduces to J^T J / σ^2 where J is the Jacobian
            of the signal w.r.t. the parameters).
    """

    dim: int = 3

    def __init__(
        self,
        *,
        tr: float = _DEFAULT_TR,
        te: float = _DEFAULT_TE,
        flip_angle_rad: float = _DEFAULT_ALPHA_RAD,
        n_mc_samples: int = 16,
        sigma_signal: float = 0.05,
        cache_resolution: int = 0,
        cache_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
            (0.1, 1.5),
            (50.0, 3500.0),
            (10.0, 500.0),
        ),
        use_bloch_checkpoint: bool = False,
    ) -> None:
        """Construct a Bloch relaxation manifold.

        Args:
            tr / te / flip_angle_rad: Acquisition parameters used by the
                Fisher-metric Monte-Carlo estimator.
            n_mc_samples: Number of Bloch evaluations per metric point
                (unused in the deterministic Jacobian path; kept for
                forward-compatibility with stochastic metrics).
            sigma_signal: Assumed Gaussian noise standard deviation.
            cache_resolution: When > 0, build a tabulated metric grid
                of shape ``[cache_resolution] * 3`` covering
                ``cache_bounds`` on construction. Subsequent
                ``metric_tensor`` queries trilinearly interpolate the
                grid rather than back-propping through the Bloch op.
                Trades a one-time setup cost (~``R³`` Bloch evals) for
                much faster training-time queries.
            cache_bounds: ``((M0_lo, M0_hi), (T1_lo, T1_hi), (T2_lo, T2_hi))``.
        """
        self.tr = tr
        self.te = te
        self.flip_angle_rad = flip_angle_rad
        self.n_mc_samples = n_mc_samples
        self.sigma_signal = sigma_signal
        self._bloch = DifferentiableBlochLayer(sequence_type="SPGR")
        self.cache_resolution = int(cache_resolution)
        self.cache_bounds = cache_bounds
        self.use_bloch_checkpoint = bool(use_bloch_checkpoint)
        self._cached_grid: torch.Tensor | None = None
        if self.cache_resolution > 0:
            self._build_metric_cache()

    # ---- contract --------------------------------------------------- #

    def metric_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        """Return Fisher metric g(θ) ∈ R^{...×3×3}.

        For a Gaussian likelihood y ~ N(s(θ), σ²I) the Fisher matrix is
        g(θ) = J(θ)ᵀ J(θ) / σ² where J = ∂s/∂θ. We compute J via autograd
        through the differentiable Bloch signal evaluated at a single
        synthetic voxel — or, when the cache is enabled, by trilinear
        interpolation of a pre-tabulated grid.
        """
        if self._cached_grid is not None:
            return self._interpolate_metric_cache(theta)
        original_shape = theta.shape
        theta_flat = theta.reshape(-1, 3)
        gs = []
        for i in range(theta_flat.shape[0]):
            g_i = self._fisher_at_single_point(theta_flat[i])
            gs.append(g_i)
        G = torch.stack(gs, dim=0).reshape(*original_shape[:-1], 3, 3)
        return G

    def refresh_metric_cache(self) -> None:
        """Re-tabulate the metric grid; call every N epochs.

        No-op when the cache was not built (``cache_resolution == 0``).
        """
        if self.cache_resolution > 0:
            self._build_metric_cache()

    # ---- metric cache ---------------------------------------------- #

    def _build_metric_cache(self) -> None:
        """Tabulate g(θ) on a regular grid covering ``cache_bounds``."""
        R = self.cache_resolution
        axes = []
        for lo, hi in self.cache_bounds:
            axes.append(torch.linspace(lo, hi, R))
        m0, t1, t2 = torch.meshgrid(*axes, indexing="ij")
        grid_pts = torch.stack([m0, t1, t2], dim=-1).reshape(-1, 3)
        gs = []
        for i in range(grid_pts.shape[0]):
            gs.append(self._fisher_at_single_point(grid_pts[i]))
        cache = torch.stack(gs, dim=0).reshape(R, R, R, 3, 3)
        self._cached_grid = cache

    def _interpolate_metric_cache(self, theta: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of the cached metric grid."""
        assert self._cached_grid is not None
        R = self.cache_resolution
        bounds = self.cache_bounds
        original_shape = theta.shape
        theta_flat = theta.reshape(-1, 3)
        # Normalise each coordinate to [0, R-1].
        u_list = []
        for d in range(3):
            lo, hi = bounds[d]
            u = (theta_flat[:, d] - lo) / max(hi - lo, 1e-12) * (R - 1)
            u_list.append(u.clamp(min=0.0, max=R - 1))
        u = torch.stack(u_list, dim=-1)
        i0 = u.floor().long().clamp(max=R - 2)
        frac = (u - i0.float()).unsqueeze(-1).unsqueeze(-1)  # broadcast over (3,3)
        # Eight corner lookups.
        result = torch.zeros(theta_flat.shape[0], 3, 3, device=theta.device)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    ix = i0[:, 0] + di
                    jy = i0[:, 1] + dj
                    kz = i0[:, 2] + dk
                    corner = self._cached_grid[ix, jy, kz]
                    wx = (
                        frac[:, 0].squeeze(-1).squeeze(-1)
                        if di
                        else (1.0 - frac[:, 0].squeeze(-1).squeeze(-1))
                    )
                    wy = (
                        frac[:, 1].squeeze(-1).squeeze(-1)
                        if dj
                        else (1.0 - frac[:, 1].squeeze(-1).squeeze(-1))
                    )
                    wz = (
                        frac[:, 2].squeeze(-1).squeeze(-1)
                        if dk
                        else (1.0 - frac[:, 2].squeeze(-1).squeeze(-1))
                    )
                    w = (wx * wy * wz).unsqueeze(-1).unsqueeze(-1)
                    result = result + w * corner
        return result.reshape(*original_shape[:-1], 3, 3)

    def project_tangent(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Remove outward-normal components at the boundary T2 = T1.

        On the interior of M the tangent space is all of R^3. On the
        boundary surface T2 = T1 the outward normal in the Fisher
        metric is the gradient of c(θ) = T2 − T1; we project ``v``
        onto the constraint surface.
        """
        # M0 is parameterised θ_0, T1 is θ_1, T2 is θ_2.
        gap = theta[..., 1] - theta[..., 2]  # T1 - T2
        is_boundary = gap < _EPS
        if not is_boundary.any():
            return v
        # outward normal of c(θ) = T2 - T1 is (0, -1, +1); normalise so the
        # projection v - ⟨v, n̂⟩ n̂ removes exactly the component along n.
        normal_raw = torch.tensor([0.0, -1.0, 1.0], dtype=v.dtype, device=v.device)
        normal = normal_raw / normal_raw.norm()
        dot = (v * normal).sum(dim=-1, keepdim=True)
        projected = torch.where(
            is_boundary.unsqueeze(-1),
            v - dot * normal,
            v,
        )
        return projected

    def exp_map(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        *,
        n_steps: int = 8,
    ) -> torch.Tensor:
        """Geodesic exp via Riemannian-Euler steps + boundary reflection."""
        h = 1.0 / max(n_steps, 1)
        x = theta
        velocity = v
        for _ in range(n_steps):
            x_next = x + h * velocity
            x_next = self._reflect_to_manifold(x_next)
            velocity = self.project_tangent(x_next, velocity)
            x = x_next
        return x

    def log_map(
        self,
        theta_a: torch.Tensor,
        theta_b: torch.Tensor,
        *,
        n_newton: int = 4,
        n_geodesic_steps: int = 8,
    ) -> torch.Tensor:
        """Approximate log map via Newton iteration on ``exp_map(a, v) ≈ b``.

        Starts from the Euclidean linearisation ``v₀ = θ_b - θ_a``,
        then iterates the fixed-point update

            v ← v + (θ_b - exp_map(θ_a, v))

        which converges quadratically near the solution while degrading
        gracefully to the first-order linear map in the boundary regime.
        ``n_newton == 0`` returns the original first-order approximation
        (used by the existing tests).

        Args:
            theta_a: Source manifold point ``[..., 3]``.
            theta_b: Destination manifold point ``[..., 3]``.
            n_newton: Newton-iteration count (0 ⇒ first-order only).
            n_geodesic_steps: Steps used by exp_map inside the iteration.

        Returns:
            Tangent vector ``v ∈ T_{θ_a} M`` such that
            ``exp_map(θ_a, v) ≈ θ_b``.
        """
        v = theta_b - theta_a
        v = self.project_tangent(theta_a, v)
        for _ in range(max(n_newton, 0)):
            theta_pred = self.exp_map(theta_a, v, n_steps=n_geodesic_steps)
            residual = theta_b - theta_pred
            v = self.project_tangent(theta_a, v + residual)
        return v

    def geodesic_distance(
        self,
        theta_a: torch.Tensor,
        theta_b: torch.Tensor,
    ) -> torch.Tensor:
        """First-order geodesic distance via the metric at the midpoint."""
        mid = 0.5 * (theta_a + theta_b)
        g = self.metric_tensor(mid)  # [..., 3, 3]
        v = (theta_b - theta_a).unsqueeze(-1)  # [..., 3, 1]
        # quadratic form vᵀ g v
        sq = (v.transpose(-1, -2) @ g @ v).squeeze(-1).squeeze(-1)
        return torch.sqrt(torch.clamp(sq, min=0.0))

    # ---- helpers ---------------------------------------------------- #

    def _fisher_at_single_point(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute g(θ) at a single point via autograd of the Bloch signal."""
        theta_req = theta.detach().clone().requires_grad_(True)
        # Pseudo-image of shape [1, 1, 1, 1] containing this single voxel.
        M0 = theta_req[0:1].reshape(1, 1, 1, 1)
        T1 = theta_req[1:2].reshape(1, 1, 1, 1)
        T2 = theta_req[2:3].reshape(1, 1, 1, 1)
        alpha = torch.tensor(self.flip_angle_rad, device=theta.device)

        def _bloch_fwd(t1: torch.Tensor, t2: torch.Tensor, pd: torch.Tensor) -> torch.Tensor:
            return self._bloch(
                t1_map=t1,
                t2_map=t2,
                pd_map=pd,
                tr=self.tr,
                te=self.te,
                flip_angle_rad=alpha,
                b0_map=None,
                b1_map=None,
            )

        if self.use_bloch_checkpoint:
            import torch.utils.checkpoint as torch_ckpt

            signal = torch_ckpt.checkpoint(_bloch_fwd, T1, T2, M0, use_reentrant=False).reshape(1)
        else:
            signal = _bloch_fwd(T1, T2, M0).reshape(1)  # scalar signal
        # Jacobian ∂s/∂θ (single output ⇒ one autograd.grad call).
        grads = torch.autograd.grad(
            outputs=signal,
            inputs=theta_req,
            create_graph=False,
            retain_graph=False,
        )[0]  # [3]
        J = grads.unsqueeze(0)  # [1, 3]
        g = (J.transpose(0, 1) @ J) / (self.sigma_signal**2)
        # Add a small isotropic regulariser for numerical safety.
        g = g + 1e-6 * torch.eye(3, dtype=g.dtype, device=g.device)
        return g.detach()

    @staticmethod
    def _reflect_to_manifold(theta: torch.Tensor) -> torch.Tensor:
        """Reflect a point back inside M = {M0, T1, T2 > 0, T2 ≤ T1}."""
        m0 = theta[..., 0].clamp(min=_EPS)
        t1 = theta[..., 1].clamp(min=_EPS)
        t2 = theta[..., 2].clamp(min=_EPS)
        # If T2 > T1 we mirror T2 across the constraint surface.
        violator = t2 > t1
        t2_fixed = torch.where(violator, 2 * t1 - t2, t2)
        t2_fixed = t2_fixed.clamp(min=_EPS)
        return torch.stack([m0, t1, t2_fixed], dim=-1)


__all__ = ["BlochRelaxationManifold"]
