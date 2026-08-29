r"""Schramm-Loewner Evolution (SLE) :math:`{}_\kappa` acquisition trajectories.

The chordal SLE :math:`{}_\kappa` curve in the upper half-plane
:math:`\mathbb{H}` is the random simple curve driven by the scaled Brownian
motion :math:`\sqrt{\kappa}B_t` through the Loewner equation

.. math::

    \partial_t g_t(z) = \frac{2}{g_t(z) - \sqrt{\kappa}B_t}.

By Schramm's theorem the law of the curve is conformally invariant; by
Beffara the Hausdorff dimension is :math:`\min(1+\kappa/8, 2)`. Sampling
k-space along an SLE :math:`{}_\kappa` curve yields fractal, conformally
invariant masks with a Talagrand-chaining RIP bound
:math:`\delta=O(s^{1-d/2}\sqrt{\log N/m})`.

This module implements the *discrete slit-mapping* construction
(Marshall-Rohde) which approximates the curve to first order in
:math:`\Delta t` by composing backward vertical slit maps
:math:`f_k(z)=U_k+\sqrt{(z-U_k)^2-\varepsilon^2}` with
:math:`\varepsilon=\sqrt{2\Delta t}` and driver
:math:`U_k=\sqrt{\kappa}B_{t_k}`. The trace is then rasterised onto the
k-space grid as a binary acquisition mask.

Consistency check: in the deterministic limit :math:`\kappa\to 0` the
driver vanishes and the slit-map composition produces a straight vertical
segment — Cartesian-line acquisition.
"""

from __future__ import annotations

import math

import torch

__all__ = ["build_sle_kspace_mask", "kappa_to_dimension", "sample_sle_trace"]


def kappa_to_dimension(kappa: float) -> float:
    r"""Hausdorff dimension :math:`\dim_H(\gamma)=\min(1+\kappa/8, 2)` (Beffara 2008)."""
    if kappa < 0.0:
        raise ValueError(f"kappa must be non-negative; got {kappa}.")
    return min(1.0 + kappa / 8.0, 2.0)


def sample_sle_trace(
    kappa: float,
    n_steps: int,
    *,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Sample one realisation of an SLE :math:`{}_\kappa` trace in :math:`\mathbb{H}`.

    Args:
        kappa: SLE parameter in :math:`[0,8)`; controls fractal dimension.
        n_steps: Number of discretisation steps in time.
        T: Final time. Curve length scales as :math:`\sqrt{T}`.
        generator: Optional :class:`torch.Generator` for reproducibility.
        device: Device on which to allocate.

    Returns:
        Tensor of shape ``(n_steps + 1, 2)`` giving ``(x, y)`` Cartesian
        coordinates of the trace tip at each time. ``y >= 0`` always.

    Notes:
        Uses the discrete slit-mapping construction
        :math:`f_k(z)=U_k+\sqrt{(z-U_k)^2-\varepsilon^2}` with
        :math:`\varepsilon=\sqrt{2\Delta t}`. The principal square root
        branch keeps the trace in the upper half-plane.
    """
    if kappa < 0.0 or kappa >= 8.0:
        raise ValueError(f"kappa must lie in [0, 8); got {kappa}.")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1.")

    dt = T / n_steps
    eps = math.sqrt(2.0 * dt)

    increments = torch.randn(n_steps, generator=generator, device=device) * math.sqrt(dt)
    bm = torch.cumsum(increments, dim=0)
    drivers = math.sqrt(kappa) * bm

    trace = torch.zeros((n_steps + 1, 2), device=device)
    z_real = torch.zeros((), device=device)
    z_imag = torch.full((), eps, device=device)

    for k in range(n_steps - 1, -1, -1):
        u = drivers[k]
        dx = z_real - u
        dy = z_imag
        s_real = dx * dx - dy * dy - eps * eps
        s_imag = 2.0 * dx * dy
        r = torch.sqrt(s_real * s_real + s_imag * s_imag)
        sqrt_real = torch.sqrt((r + s_real).clamp_min(0.0) / 2.0)
        sqrt_imag = torch.sign(s_imag).clamp_min(0.0) * torch.sqrt(
            (r - s_real).clamp_min(0.0) / 2.0
        ) + (1.0 - torch.sign(s_imag).abs()) * torch.sqrt((r - s_real).clamp_min(0.0) / 2.0)
        # Ensure positive imaginary part (principal branch in H).
        sqrt_imag = sqrt_imag.abs()
        z_real = u + sqrt_real
        z_imag = sqrt_imag
        trace[k] = torch.stack([z_real, z_imag])

    trace[n_steps] = trace[n_steps - 1].clone()
    return trace


def build_sle_kspace_mask(
    kappa: float,
    shape: tuple[int, int],
    n_steps: int,
    *,
    T: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    r"""Rasterise an SLE :math:`{}_\kappa` trace into a binary k-space mask.

    The trace is centred and scaled into the image grid; lattice points
    visited by the discretised trace receive value 1, others 0. The mask
    always contains the DC sample at the grid centre.

    Args:
        kappa: SLE parameter in :math:`[0,8)`.
        shape: ``(H, W)`` of the k-space grid.
        n_steps: Number of trace points to draw (recommend :math:`O(\sqrt{HW})`).
        T: Loewner final time.
        generator: Optional :class:`torch.Generator`.
        device: Device on which to allocate the mask.

    Returns:
        Binary mask of dtype ``torch.bool`` and shape ``shape``.
    """
    h, w = shape
    trace = sample_sle_trace(kappa=kappa, n_steps=n_steps, T=T, generator=generator, device=device)

    # Normalise: y in [0, max_y] -> [0, h], x in [min_x, max_x] -> [0, w].
    x = trace[:, 0]
    y = trace[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    eps_norm = 1e-6
    x_range = (x_max - x_min).clamp_min(eps_norm)
    y_range = (y_max - y_min).clamp_min(eps_norm)
    x_idx = ((x - x_min) / x_range * (w - 1)).long()
    y_idx = ((y - y_min) / y_range * (h - 1)).long()
    x_idx = x_idx.clamp(0, w - 1)
    y_idx = y_idx.clamp(0, h - 1)

    mask = torch.zeros(shape, dtype=torch.bool, device=x_idx.device)
    mask[y_idx, x_idx] = True
    mask[h // 2, w // 2] = True
    return mask
