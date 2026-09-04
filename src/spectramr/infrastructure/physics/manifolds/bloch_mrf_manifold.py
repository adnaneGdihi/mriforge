"""5-D Bloch parameter manifold for MRF posterior inference (MRF §2).

Manifold:

.. math::
    \\mathcal{M}_{\\text{MRF}} = \\{(T_1, T_2, M_0, B_0, B_1) :
    T_1>0,\\, T_2>0,\\, T_2 \\le T_1,\\, M_0>0,\\, B_1>0\\}
    \\subset \\mathbb{R}^5.

Equipped with the Fisher information metric of the Bloch likelihood
under a given MRF pulse sequence. Two practical contributions of
this module:

* :class:`BlochMRFManifold` exposes ``metric_tensor``, ``exp_map``,
  ``log_map``, ``project_tangent`` matching the
  :class:`RiemannianManifold` protocol so it can be plugged into the
  existing Riemannian-diffusion strategy.
* :func:`cayley_hilbert_chart` / :func:`inverse_cayley_hilbert_chart`
  send the manifold to a half-space in :math:`\\mathbb{R}^5` where
  the relaxation-ordering constraint becomes trivial. The chart is
  conformal at the manifold's centre.

The Fisher metric is computed via Hutchinson trace estimation
through ``differentiable_bloch`` ([1, §3]); we ship a lightweight
Monte-Carlo estimator that doesn't require importing the heavy
Bloch operator at construction time.

References:
    [1] Y. Jiang et al., "MR fingerprinting using FISP with spiral readout",
        *Magn. Reson. Med.*, 74(6), 2015.
    [11] S.-i. Amari & H. Nagaoka, *Methods of Information Geometry*, AMS, 2000.
    [12] V. De Bortoli et al., "Riemannian score-based generative modelling",
         *NeurIPS*, 2022.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

EPS = 1e-6


def cayley_hilbert_chart(theta: torch.Tensor) -> torch.Tensor:
    """Bijection ``(T1, T2, M0, B0, B1) → (logT1, log(T1-T2), logM0, B0, logB1)``.

    Sends the manifold's open interior to ``ℝ^5``; the relaxation
    constraint ``T2 ≤ T1`` becomes the trivial requirement that the
    second coordinate is finite (it is unbounded above and below in
    the chart).
    """
    if theta.shape[-1] != 5:
        raise ValueError(f"expected 5-D theta, got shape {theta.shape}")
    T1, T2, M0, B0, B1 = theta.unbind(-1)
    return torch.stack(
        [
            T1.clamp_min(EPS).log(),
            (T1 - T2).clamp_min(EPS).log(),
            M0.clamp_min(EPS).log(),
            B0,
            B1.clamp_min(EPS).log(),
        ],
        dim=-1,
    )


def inverse_cayley_hilbert_chart(z: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`cayley_hilbert_chart`."""
    if z.shape[-1] != 5:
        raise ValueError(f"expected 5-D chart point, got shape {z.shape}")
    log_T1, log_T1_minus_T2, log_M0, B0, log_B1 = z.unbind(-1)
    T1 = log_T1.exp()
    T2 = T1 - log_T1_minus_T2.exp()
    return torch.stack([T1, T2, log_M0.exp(), B0, log_B1.exp()], dim=-1)


class BlochMRFManifold(torch.nn.Module):
    """Bloch parameter manifold with Fisher metric and Cayley-Hilbert chart.

    Args:
        bloch_simulator: Callable mapping ``(theta[..., 5], pulse_sequence)``
            to fingerprint ``[..., N]``. Default is a tiny synthetic
            stand-in suitable for unit tests; production callers
            inject the real ``differentiable_bloch`` operator.
        n_hutchinson: Number of Hutchinson probes for the Fisher trace
            estimator.
    """

    dim: int = 5

    def __init__(
        self,
        bloch_simulator: Callable[[torch.Tensor], torch.Tensor] | None = None,
        n_hutchinson: int = 4,
    ) -> None:
        super().__init__()
        self._sim = bloch_simulator or self._toy_bloch
        self.n_hutchinson = n_hutchinson

    @staticmethod
    def _toy_bloch(theta: torch.Tensor) -> torch.Tensor:
        """Toy fingerprint: ``f(θ) = exp(-t/T2) * (1 - exp(-t/T1))`` on
        a fixed grid. Closed form; useful for tests where the heavy
        differentiable_bloch operator is not needed."""
        T1, T2, M0, _B0, _B1 = theta.unbind(-1)
        t = torch.linspace(0.0, 1.0, 32, device=theta.device, dtype=T1.dtype)
        amp = M0.unsqueeze(-1) * (1.0 - (-t / T1.unsqueeze(-1).clamp_min(EPS)).exp())
        decay = (-t / T2.unsqueeze(-1).clamp_min(EPS)).exp()
        return amp * decay

    def metric_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        """Fisher information metric at ``theta`` (unit noise variance).

        Computes ``g_ij(θ) = Σ_n ∂f_n/∂θ_i · ∂f_n/∂θ_j = (JᵀJ)_ij`` with the
        full per-output Jacobian ``J[..., n, i] = ∂f_n/∂θ_i``.

        The previous implementation looped ``dim`` times over the SAME summed
        gradient ``∂(Σ_n f_n)/∂θ`` (the loop index was unused), giving ``dim``
        identical columns and an outer product of shape ``[..., dim, dim, dim]``
        — a meaningless, wrongly-shaped tensor (the old comment admitted it).
        Each output component is differentiated separately here; summing the
        scalar ``f[..., n].sum()`` over the batch is exact per-sample because
        ``f[b]`` depends only on ``θ[b]``.
        """
        theta_in = theta.detach().clone().requires_grad_(True)
        f = self._sim(theta_in)  # [..., N]
        n_out = f.shape[-1]
        rows = [
            torch.autograd.grad(f[..., n].sum(), theta_in, retain_graph=True, create_graph=False)[0]
            for n in range(n_out)
        ]
        jac = torch.stack(rows, dim=-2)  # [..., N, dim]
        g = jac.transpose(-1, -2) @ jac  # [..., dim, dim]
        eye = torch.eye(self.dim, device=theta.device, dtype=theta.dtype)
        return g + EPS * eye

    def exp_map(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        *,
        n_steps: int = 4,
    ) -> torch.Tensor:
        """Approximate geodesic via Cayley-Hilbert Euler integration.

        We integrate in the flat chart and pull back, which is the
        first-order correct geodesic when the chart is conformal at
        the integration point — true at the manifold centre and
        approximately true elsewhere.
        """
        z = cayley_hilbert_chart(theta)
        for _ in range(n_steps):
            z = z + v / n_steps
        return inverse_cayley_hilbert_chart(z)

    def log_map(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Chart-space displacement (first-order log map)."""
        return cayley_hilbert_chart(theta_b) - cayley_hilbert_chart(theta_a)

    def project_tangent(self, theta: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Project ``v`` to a valid tangent direction.

        Reflects the velocity component pointing across the
        ``T_2 = T_1`` boundary (i.e. where chart coordinate 1 → -∞).
        """
        z = cayley_hilbert_chart(theta)
        near_boundary = z[..., 1] < -4.0  # log(T1 - T2) very small
        v = v.clone()
        v[..., 1] = torch.where(near_boundary, v[..., 1].clamp_min(0.0), v[..., 1])
        return v


__all__ = [
    "BlochMRFManifold",
    "cayley_hilbert_chart",
    "inverse_cayley_hilbert_chart",
]
