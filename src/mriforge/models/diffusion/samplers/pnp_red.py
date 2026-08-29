"""Plug-and-Play (PnP) and Regularization-by-Denoising (RED) priors.

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-9 (M1).

Two related image-recovery schemes that use a generic image
denoiser ``D`` as an implicit prior:

- **PnP-ADMM**: solves ``min_x  ½ ‖A x − y‖² + λ R(x)`` with ADMM where
  the prior-proximal step is replaced by a single denoiser call:
  ``x = D(z − u)``.
- **RED**: uses the regulariser ``R(x) = ½ x^T (x − D(x))`` whose
  gradient is ``∇R(x) = x − D(x)``.  Plug into any first-order method.

Both are registered under :mod:`mriforge.models.diffusion.samplers.registry`
so they can be selected via the standard sampler factory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .registry import register_sampler


@dataclass
class PnPADMMConfig:
    n_iters: int = 30
    rho: float = 1.0
    lambda_prior: float = 0.1
    cg_iters: int = 5


@register_sampler("pnp_admm", aliases=["PnPADMM", "PlugAndPlay"])
class PnPADMMSampler:
    """Plug-and-Play ADMM for ``min ½‖Ax − y‖² + λ R(x)``."""

    def __init__(
        self, n_iters: int = 30, rho: float = 1.0, lambda_prior: float = 0.1, cg_iters: int = 5
    ) -> None:
        self.cfg = PnPADMMConfig(
            n_iters=n_iters, rho=rho, lambda_prior=lambda_prior, cg_iters=cg_iters
        )

    def sample(
        self,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """ADMM splitting:
        x ← (Aᵀ A + ρ I)⁻¹ (Aᵀ y + ρ (z − u))    [via CG]
        z ← D(x + u)
        u ← u + x − z
        """
        x = torch.zeros(*shape, device=device)
        z = x.clone()
        u = torch.zeros_like(x)
        Aty = adjoint_op(y)

        def AtA(v):
            return adjoint_op(forward_op(v)) + self.cfg.rho * v

        for _ in range(self.cfg.n_iters):
            rhs = Aty + self.cfg.rho * (z - u)
            # CG inner solve for (AᵀA + ρ I) x = rhs
            xk = x.clone()
            r = rhs - AtA(xk)
            p = r.clone()
            r_norm_sq = (r * r).sum()
            for _ in range(self.cfg.cg_iters):
                Ap = AtA(p)
                alpha = r_norm_sq / (p * Ap).sum().clamp_min(1e-12)
                xk = xk + alpha * p
                r_new = r - alpha * Ap
                r_new_norm_sq = (r_new * r_new).sum()
                beta = r_new_norm_sq / r_norm_sq.clamp_min(1e-12)
                p = r_new + beta * p
                r = r_new
                r_norm_sq = r_new_norm_sq
            x = xk
            z = denoiser(x + u)
            u = u + x - z
        return z


@dataclass
class REDConfig:
    n_iters: int = 50
    step_size: float = 0.05
    lambda_prior: float = 0.1


@register_sampler("red", aliases=["RED", "RegularizationByDenoising"])
class REDSampler:
    """Regularization-by-Denoising via gradient descent.

    Update: ``x ← x − η (Aᵀ(A x − y) + λ (x − D(x)))``.
    """

    def __init__(
        self, n_iters: int = 50, step_size: float = 0.05, lambda_prior: float = 0.1
    ) -> None:
        self.cfg = REDConfig(n_iters=n_iters, step_size=step_size, lambda_prior=lambda_prior)

    def sample(
        self,
        denoiser: Callable[[torch.Tensor], torch.Tensor],
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        x = adjoint_op(y)
        for _ in range(self.cfg.n_iters):
            data_grad = adjoint_op(forward_op(x) - y)
            prior_grad = x - denoiser(x)
            x = x - self.cfg.step_size * (data_grad + self.cfg.lambda_prior * prior_grad)
        return x


__all__ = ["PnPADMMConfig", "PnPADMMSampler", "REDConfig", "REDSampler"]
