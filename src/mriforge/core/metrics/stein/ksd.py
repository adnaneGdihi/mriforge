"""Kernelised Stein Discrepancy (KSD) — Liu, Lee & Jordan (ICML 2016).

The KSD measures how well an unnormalised model ``p_θ`` fits an empirical
sample ``{x_i}_{i=1}^n``. Given access only to the score function
``s_θ(x) = ∇_x log p_θ(x)`` it is computable in closed form as a
U-statistic over a positive-definite kernel.

For samples ``x_i`` (i.i.d.) and kernel ``k`` the Stein kernel is

    k_p(x, y) = sᵀ(x) k(x,y) s(y)
              + sᵀ(x) ∇_y k(x,y)
              + (∇_x k(x,y))ᵀ s(y)
              + tr(∇_x ∇_y k(x,y))

and the unbiased U-statistic estimator of ``KSD²`` is

    KSD²(P_θ ‖ Q_n) = 1/(n(n-1)) ∑_{i≠j} k_p(x_i, x_j).

We use the inverse-multi-quadric kernel
``k(x,y) = (c² + ‖x-y‖²)^{-1/2}`` with the median heuristic ``c =
median{‖x_i - x_j‖}`` (Chwialkowski et al. 2016 — proves detection
consistency).

p-values are produced by the wild bootstrap of Chwialkowski et al. (2016).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class KSDResult:
    """Output of a single KSD evaluation."""

    ksd_squared: float
    p_value: float | None
    n_samples: int
    kernel: str
    bandwidth: float


def _pairwise_sq_dist(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances ‖x_i - x_j‖² for x of shape [n, d]."""
    return torch.cdist(x, x, p=2.0).pow(2)


def _median_heuristic(x: torch.Tensor) -> float:
    """Median pairwise distance over off-diagonal pairs (Garreau et al. 2018)."""
    n = x.shape[0]
    d2 = _pairwise_sq_dist(x)
    # Upper triangle excluding diagonal.
    iu = torch.triu_indices(n, n, offset=1, device=x.device)
    distances = d2[iu[0], iu[1]].sqrt()
    return distances.median().item() if distances.numel() > 0 else 1.0


def _imq_stein_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    c: float,
    beta: float = -0.5,
) -> torch.Tensor:
    """Stein kernel for the inverse-multi-quadric base kernel.

    k(x, y) = (c² + ‖x-y‖²)^β with β = -1/2 by default.

    The closed form for the Stein kernel is given by Eq. (22) of
    Gorham & Mackey (NeurIPS 2017).

    Args:
        x, y: [n, d], [m, d] sample tensors.
        sx, sy: [n, d], [m, d] score vectors at each sample.
        c: kernel bandwidth.
        beta: kernel exponent.

    Returns:
        [n, m] matrix of Stein-kernel values.
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [n, m, d]
    sq = (diff * diff).sum(dim=-1)  # [n, m]
    base = (c * c + sq).pow(beta)  # k(x,y)
    # ∇_x k(x,y) = 2β diff (c²+‖·‖²)^{β-1} — vector in d.
    base_dx = 2.0 * beta * (c * c + sq).pow(beta - 1)  # [n, m] scalar prefactor
    grad_x_k = base_dx.unsqueeze(-1) * diff  # ∇_x k, [n, m, d]
    grad_y_k = -grad_x_k  # ∇_y k = -∇_x k
    # tr(∇_x ∇_y k) = -2β d (c²+‖·‖²)^{β-1} - 4β(β-1) ‖diff‖² (c²+‖·‖²)^{β-2}
    d = x.shape[-1]
    trace_term = -2.0 * beta * d * (c * c + sq).pow(beta - 1) - 4.0 * beta * (beta - 1) * sq * (
        c * c + sq
    ).pow(beta - 2)
    term_ss = (sx.unsqueeze(1) * sy.unsqueeze(0)).sum(dim=-1) * base  # sᵀs k
    term_sgy = (sx.unsqueeze(1) * grad_y_k).sum(dim=-1)
    term_sxg = (grad_x_k * sy.unsqueeze(0)).sum(dim=-1)
    return term_ss + term_sgy + term_sxg + trace_term


def kernelised_stein_discrepancy(
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    samples: torch.Tensor,
    *,
    bandwidth: float | None = None,
    kernel: str = "imq",
    return_kernel_matrix: bool = False,
) -> tuple[float, torch.Tensor | None]:
    """Compute the unbiased U-statistic estimator of KSD².

    Args:
        score_fn: Callable that takes [n, d] samples and returns [n, d]
            score vectors ∇_x log p_θ(x).
        samples: [n, d] tensor of i.i.d. samples to test for fit.
        bandwidth: Optional kernel bandwidth. If None, set by the
            median heuristic.
        kernel: Currently only "imq" supported.
        return_kernel_matrix: If True, also return the [n, n] Stein-kernel
            matrix (needed by the wild-bootstrap p-value routine).

    Returns:
        Tuple ``(ksd_squared, kernel_matrix_or_None)``.
    """
    if kernel != "imq":
        raise ValueError(f"Only 'imq' kernel supported, got {kernel}")
    if samples.dim() != 2:
        raise ValueError(f"samples must be 2-D [n, d], got shape {tuple(samples.shape)}")
    n = samples.shape[0]
    if n < 2:
        raise ValueError("KSD requires at least 2 samples")

    c = bandwidth if bandwidth is not None else _median_heuristic(samples)
    s = score_fn(samples)
    K = _imq_stein_kernel(samples, samples, s, s, c=c)
    # Zero out diagonal for the U-statistic.
    K_off = K.clone()
    K_off.fill_diagonal_(0.0)
    ksd2 = K_off.sum() / (n * (n - 1))
    return float(ksd2.item()), K if return_kernel_matrix else None


def ksd_p_value(
    kernel_matrix: torch.Tensor,
    *,
    n_bootstrap: int = 1000,
    generator: torch.Generator | None = None,
) -> float:
    """Wild-bootstrap p-value for KSD² ≤ 0 null (Chwialkowski et al. 2016).

    Args:
        kernel_matrix: [n, n] Stein-kernel matrix from
            ``kernelised_stein_discrepancy(..., return_kernel_matrix=True)``.
        n_bootstrap: Number of bootstrap replications.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Empirical p-value in [0, 1]. Small p ⇒ reject null
        (i.e. model is a poor fit).
    """
    n = kernel_matrix.shape[0]
    K = kernel_matrix.clone()
    K.fill_diagonal_(0.0)
    observed = K.sum() / (n * (n - 1))
    count_ge = 0
    for _ in range(n_bootstrap):
        w = (
            2.0
            * torch.randint(
                0,
                2,
                (n,),
                device=K.device,
                dtype=K.dtype,
                generator=generator,
            )
            - 1.0
        )
        boot = (w.unsqueeze(1) * w.unsqueeze(0) * K).sum() / (n * (n - 1))
        if boot.item() >= observed.item():
            count_ge += 1
    return (count_ge + 1) / (n_bootstrap + 1)


__all__ = [
    "KSDResult",
    "kernelised_stein_discrepancy",
    "ksd_p_value",
]
