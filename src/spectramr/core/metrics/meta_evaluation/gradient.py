"""Metric-gradient utilities for SFA.

A metric ``M_k(x, hat_x)`` may be:

* fully differentiable (most distortion metrics implemented in torch ops),
* partially differentiable (SSIM with custom kernels, LPIPS),
* non-differentiable (Dice on argmax-thresholded segmentation, Wilcoxon).

For SFA we need ``∇_{hat_x} M_k(x, hat_x)``. ``metric_gradient`` picks the
right route per metric:

1. Try ``torch.autograd.grad`` with ``hat_x`` as a leaf — if it produces a
   finite tensor, return it.
2. Otherwise fall back to a Gaussian-perturbation SPSA estimator (SPSA-GS),
   averaged over ``n_samples`` Rademacher / Gaussian directions.

The fallback is on by default because most "perceptual" metrics in the
codebase wrap CPU/numpy ops that break autograd silently.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def autograd_gradient(
    metric: Callable[..., torch.Tensor | float],
    clean: torch.Tensor,
    hat: torch.Tensor,
    context: object | None = None,
) -> torch.Tensor | None:
    """Try autograd; return ``None`` if it fails or yields NaN."""
    kw = {"context": context} if context is not None else {}
    try:
        hat_leaf = hat.detach().clone().float().requires_grad_(True)
        value = metric(hat_leaf, clean, **kw)
        if isinstance(value, float):
            return None
        if not isinstance(value, torch.Tensor):
            return None
        if not value.requires_grad:
            return None
        scalar = value if value.ndim == 0 else value.mean()
        grad = torch.autograd.grad(scalar, hat_leaf, retain_graph=False, allow_unused=True)[0]
        if grad is None:
            return None
        if not torch.isfinite(grad).all():
            return None
        return grad.detach()
    except Exception:
        return None


def spsa_gradient(
    metric: Callable[..., float],
    clean: torch.Tensor,
    hat: torch.Tensor,
    delta: float = 1e-2,
    n_samples: int = 8,
    seed: int = 0,
    context: object | None = None,
) -> torch.Tensor:
    """Gaussian-perturbation simultaneous stochastic gradient.

    Estimator: ``g ≈ (1/S) Σ_s ((M(x + δ u_s) - M(x − δ u_s)) / 2δ) u_s``
    with ``u_s ~ N(0, I)``. Has bounded variance independent of ``dim(x)``
    when ``M`` is L-smooth, which is sufficient for our cosine alignment.

    The gradient is returned on ``hat.device``. Perturbation directions are
    drawn from a CPU generator seeded by ``seed`` and moved onto the device,
    so the estimate is reproducible *and* identical whether ``hat`` is on CPU
    or CUDA.
    """
    kw = {"context": context} if context is not None else {}
    gen = torch.Generator().manual_seed(seed)
    g = torch.zeros_like(hat.float())
    for s in range(n_samples):
        # Draw directions on CPU then move to hat.device: keeps the RNG
        # stream device-independent (same ranking on CPU and GPU) and avoids
        # the cross-device add when hat lives on CUDA.
        u = torch.randn(hat.shape, generator=gen, dtype=torch.float32).to(hat.device)
        plus = (hat.float() + delta * u).to(hat.dtype)
        minus = (hat.float() - delta * u).to(hat.dtype)
        try:
            v_plus = float(metric(plus, clean, **kw))
            v_minus = float(metric(minus, clean, **kw))
        except Exception:
            return torch.zeros_like(hat.float())
        diff = (v_plus - v_minus) / (2 * delta)
        if not (diff == diff):  # NaN
            return torch.zeros_like(hat.float())
        g = g + diff * u
    return (g / max(n_samples, 1)).to(hat.dtype)


def metric_gradient(
    metric: Callable[..., torch.Tensor | float],
    clean: torch.Tensor,
    hat: torch.Tensor,
    differentiable_hint: bool = True,
    delta: float = 1e-2,
    n_samples: int = 8,
    seed: int = 0,
    context: object | None = None,
) -> torch.Tensor:
    """Single entry point: autograd then SPSA fallback.

    ``context`` (the no-reference measurement bundle) is forwarded to the metric
    so NR/physics metrics are differentiated on their true acquisition; without
    it they return NaN and the SFA alignment collapses to 0.
    """
    if differentiable_hint:
        g = autograd_gradient(metric, clean, hat, context=context)
        if g is not None:
            return g
    return spsa_gradient(metric, clean, hat, delta, n_samples, seed, context=context)
