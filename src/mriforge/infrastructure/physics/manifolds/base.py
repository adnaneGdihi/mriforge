"""Riemannian manifold protocol.

This protocol is intentionally minimal: every manifold consumer in this
codebase only needs ``metric_tensor`` (for inner products), ``exp_map``
(for forward integration on the manifold), ``log_map`` (for computing
geodesic displacements), and ``project_tangent`` (for keeping
network-predicted scores in the tangent space).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RiemannianManifold(Protocol):
    """Abstract Riemannian manifold contract."""

    dim: int  # Intrinsic dimension.

    def metric_tensor(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the metric tensor g(θ) of shape [..., dim, dim].

        Args:
            theta: Point on the manifold, shape [..., dim].
        """
        ...

    def exp_map(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
        *,
        n_steps: int = 8,
    ) -> torch.Tensor:
        """Geodesic exponential: integrate the velocity ``v`` at ``theta`` for unit time.

        Args:
            theta: Base point, shape [..., dim].
            v: Tangent vector at theta, shape [..., dim].
            n_steps: Number of geodesic-ODE integrator steps.

        Returns:
            New point on the manifold, shape [..., dim].
        """
        ...

    def log_map(self, theta_a: torch.Tensor, theta_b: torch.Tensor) -> torch.Tensor:
        """Inverse of ``exp_map`` — return tangent v with exp_map(a, v) ≈ b.

        Implementations may approximate (e.g. by Newton iteration on a
        local chart) when no closed form exists.
        """
        ...

    def project_tangent(
        self,
        theta: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Project a raw direction ``v`` onto the tangent space at ``theta``.

        For interior points this is the identity; near the boundary it
        removes the component along the outward normal in the manifold
        metric.
        """
        ...


__all__ = ["RiemannianManifold"]
