r"""B₀ Mapping from Phase-Difference Images.

Implements the B₀ field estimation pipeline from Koolstra et al. (2021) §2.6:

.. math::

    \varphi(\mathbf{x}) = -2\pi \, t_{\text{shift}} \,
    \Delta B_0(\mathbf{x}) + C(\mathbf{x})

where :math:`\varphi` is the phase difference between two images acquired
with different readout time-shifts, and :math:`C` accounts for receive-phase
offsets common to both acquisitions.

The ΔB₀ map is estimated by solving a **Tikhonov-regularised** problem:

.. math::

    \hat{\mathbf{b}} = \arg\min_{\mathbf{b}} \left\{
        \|\mathbf{A}\mathbf{s} - \boldsymbol{\Phi}\|_2^2
        + \frac{\gamma}{2} \bigl(
            \|\nabla_x \mathbf{b}\|_2^2
            + \|\nabla_y \mathbf{b}\|_2^2
        \bigr)
    \right\}

The regularisation smooths the field estimate in low-SNR regions.

Reference:
    Koolstra K et al., MAGMA (2021) 34:631–642, §2.6.
"""

from __future__ import annotations

import math

import torch


def phase_difference(
    image_unshifted: torch.Tensor,
    image_shifted: torch.Tensor,
) -> torch.Tensor:
    """Compute phase difference between two complex images.

    Before computing the phase difference, the global phase of the
    shifted image is centred around zero to prevent phase wrapping
    in the case of strong static phase offsets (paper §2.3).

    Args:
        image_unshifted: Complex image from unshifted acquisition [1, 1, H, W].
        image_shifted: Complex image from shifted acquisition [1, 1, H, W].

    Returns:
        Phase difference map [1, 1, H, W] in radians.
    """
    # Static phase correction — centre BOTH images' global phase around
    # zero so that identical inputs produce a (numerically) zero
    # difference. The previous version only centred the shifted side,
    # which left a uniform offset of ``-median(angle)`` on identical
    # inputs (see test_phase_difference_identical_inputs_near_zero).
    # When both static phases match (the canonical linear-phase case)
    # the per-image medians cancel inside the conjugate product and
    # the recovered constant offset is unchanged.
    median_unshifted = torch.angle(image_unshifted).median()
    median_shifted = torch.angle(image_shifted).median()
    image_unshifted_corrected = image_unshifted * torch.exp(-1j * median_unshifted)
    image_shifted_corrected = image_shifted * torch.exp(-1j * median_shifted)

    # Phase difference via conjugate product
    # angle(conj(img1) * img2) = angle(img2) - angle(img1)
    phase_diff = torch.angle(torch.conj(image_unshifted_corrected) * image_shifted_corrected)

    return phase_diff


def _spatial_gradient_2d(
    field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute spatial gradients (finite differences) in x and y.

    Args:
        field: 2D field map [H, W].

    Returns:
        (grad_x, grad_y) each [H, W].
    """
    gx = torch.zeros_like(field)
    gy = torch.zeros_like(field)

    # Forward differences
    gx[:, :-1] = field[:, 1:] - field[:, :-1]
    gy[:-1, :] = field[1:, :] - field[:-1, :]

    return gx, gy


def _divergence_2d(
    gx: torch.Tensor,
    gy: torch.Tensor,
) -> torch.Tensor:
    """Compute negative divergence (adjoint of gradient).

    Args:
        gx, gy: Gradient components [H, W].

    Returns:
        Divergence [H, W].
    """
    div = torch.zeros_like(gx)

    # Backward differences (adjoint of forward differences)
    div[:, 1:] += gx[:, :-1]
    div[:, :-1] -= gx[:, :-1]
    div[1:, :] += gy[:-1, :]
    div[:-1, :] -= gy[:-1, :]

    return div


def estimate_b0_tikhonov(
    phase_diff: torch.Tensor,
    t_shift: float,
    gamma: float = 2e-10,
    mask: torch.Tensor | None = None,
    max_cg_iterations: int = 100,
    cg_tolerance: float = 1e-6,
) -> torch.Tensor:
    r"""Estimate ΔB₀ from a phase-difference map using Tikhonov regularisation.

    Solves:

    .. math::

        \hat{\mathbf{b}} = \arg\min_{\mathbf{b}} \left\{
            \| \mathbf{b} \cdot (-2\pi t_{\text{shift}}) - \varphi \|_2^2
            + \frac{\gamma}{2}
            (\|\nabla_x \mathbf{b}\|_2^2 + \|\nabla_y \mathbf{b}\|_2^2)
        \right\}

    This is equivalent to solving the linear system:

    .. math::

        \bigl( \alpha^2 I - \gamma \nabla^2 \bigr) \mathbf{b}
        = \alpha \, \varphi

    where :math:`\alpha = -2\pi t_{\text{shift}}` and :math:`\nabla^2` is
    the 2D Laplacian (divergence of gradient). Solved using CG.

    Args:
        phase_diff: Phase-difference map [1, 1, H, W] in radians.
        t_shift: Readout time-shift in seconds.
        gamma: Tikhonov regularisation parameter.
            - 2e-10 for simulations
            - 2e-8 for phantom measurements
            - 4e-8 for in vivo measurements
        mask: Optional binary mask [1, 1, H, W] weighting the data term.
        max_cg_iterations: Maximum CG iterations.
        cg_tolerance: CG convergence tolerance.

    Returns:
        Estimated ΔB₀ map [1, 1, H, W] in Hz.
    """
    phi = phase_diff.squeeze()  # [H, W]
    H, W = phi.shape
    device = phi.device

    alpha = -2.0 * math.pi * t_shift

    # Mask weighting (1.0 everywhere if no mask)
    if mask is not None:
        w = mask.squeeze().float()
    else:
        w = torch.ones(H, W, device=device)

    # Right-hand side: alpha * w * phi
    rhs = alpha * w * phi

    # A operator: (alpha^2 * w + gamma * (-Laplacian)) applied to b
    def apply_A(b: torch.Tensor) -> torch.Tensor:
        data_term = alpha**2 * w * b

        # Laplacian via gradient + divergence
        gx, gy = _spatial_gradient_2d(b)
        laplacian = _divergence_2d(gx, gy)
        reg_term = -gamma * laplacian

        return data_term + reg_term

    # CG solve
    b = torch.zeros(H, W, device=device)
    r = rhs - apply_A(b)
    p = r.clone()
    rs_old = torch.sum(r * r)

    for _ in range(max_cg_iterations):
        Ap = apply_A(p)
        pAp = torch.sum(p * Ap)
        if pAp.abs() < 1e-30:
            break

        step = rs_old / pAp
        b = b + step * p
        r = r - step * Ap
        rs_new = torch.sum(r * r)

        if rs_new.sqrt() < cg_tolerance:
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return b.unsqueeze(0).unsqueeze(0)


def map_b0(
    image_unshifted: torch.Tensor,
    image_shifted: torch.Tensor,
    t_shift: float,
    gamma: float = 2e-10,
    mask: torch.Tensor | None = None,
    max_cg_iterations: int = 100,
    cg_tolerance: float = 1e-6,
) -> torch.Tensor:
    """Full B₀ mapping pipeline: phase difference → Tikhonov estimation.

    Convenience function combining :func:`phase_difference` and
    :func:`estimate_b0_tikhonov`. The CG knobs are forwarded rather than pinned
    to their defaults — a wrapper that swallows the solver budget silently runs
    100 iterations no matter what the caller asked for.

    Args:
        image_unshifted: Complex image [1, 1, H, W] from unshifted acquisition.
        image_shifted: Complex image [1, 1, H, W] from shifted acquisition.
        t_shift: Readout time-shift in seconds.
        gamma: Tikhonov parameter.
        mask: Optional mask [1, 1, H, W].
        max_cg_iterations: Maximum CG iterations.
        cg_tolerance: CG convergence tolerance.

    Returns:
        Estimated ΔB₀ [1, 1, H, W] in Hz.
    """
    phi = phase_difference(image_unshifted, image_shifted)
    return estimate_b0_tikhonov(
        phi,
        t_shift,
        gamma=gamma,
        mask=mask,
        max_cg_iterations=max_cg_iterations,
        cg_tolerance=cg_tolerance,
    )


__all__ = [
    "estimate_b0_tikhonov",
    "map_b0",
    "phase_difference",
]
