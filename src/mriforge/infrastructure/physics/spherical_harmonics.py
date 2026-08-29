r"""Spherical Harmonic Fitting for B₀ Field Maps.

Provides utilities to fit B₀ field maps to a basis of real spherical
harmonics and to extrapolate the fitted field outside the object support.

Used in the Koolstra et al. (2021) framework for two purposes:

1. **Ground-truth preparation**: Fitting measured Halbach magnet fields
   to 15th-order SH to remove measurement noise.
2. **Iterative reconstruction**: Fitting reconstructed ΔB₀ maps to
   2nd-order SH after each iteration to obtain a smooth field estimate
   that extends outside the object support.

For 2D slices the relevant real spherical harmonic basis functions
(evaluated at in-plane coordinates x, y) up to order L are the
monomials:

.. math::

    \{1,\; x,\; y,\; x^2,\; xy,\; y^2,\; x^3, \ldots \}

This is the **polynomial basis** that corresponds to the projection of
3D spherical harmonics onto a fixed z-plane, which is the standard
approach used in MRI shimming.

Reference:
    Koolstra K et al., MAGMA (2021) 34:631–642, §2.6.
"""

from __future__ import annotations

import torch


def _build_2d_polynomial_basis(
    shape: tuple[int, int],
    max_order: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a 2D polynomial basis matrix up to a given order.

    The basis consists of monomials x^p * y^q where p + q <= max_order,
    evaluated on a normalised [-1, 1]² grid.

    Args:
        shape: (H, W) spatial dimensions.
        max_order: Maximum polynomial order (e.g., 2 for quadratic).
        device: Target device.

    Returns:
        Basis matrix [M, K] where M = H*W, K = number of basis functions.
    """
    H, W = shape
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx_flat = xx.reshape(-1)
    yy_flat = yy.reshape(-1)

    columns = []
    for order in range(max_order + 1):
        for p in range(order + 1):
            q = order - p
            col = (xx_flat**p) * (yy_flat**q)
            columns.append(col)

    return torch.stack(columns, dim=1)  # [M, K]


def fit_spherical_harmonics(
    b0_map: torch.Tensor,
    max_order: int = 2,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fit a B₀ map to a 2D polynomial (spherical harmonic) basis.

    Solves the least-squares problem:

    .. math::

        \hat{\mathbf{c}} = \arg\min_{\mathbf{c}}
        \| \mathbf{B} \mathbf{c} - \mathbf{b}_0 \|_2^2

    where **B** is the polynomial basis matrix and **b₀** is the measured
    field vector, optionally restricted to a binary mask (object support).

    Args:
        b0_map: Input field map [1, 1, H, W] or [H, W] in Hz.
        max_order: Maximum polynomial order (2 = quadratic, 15 = full).
        mask: Optional binary mask [1, 1, H, W] or [H, W] restricting
            the fit to voxels inside the object.

    Returns:
        (fitted_map, coefficients):
            - fitted_map: Smooth field [1, 1, H, W] evaluated everywhere
              (including outside the mask — this is the extrapolation).
            - coefficients: SH coefficient vector [K].
    """
    # Normalise input shape
    if b0_map.dim() == 4:
        b0_2d = b0_map.squeeze(0).squeeze(0)
    elif b0_map.dim() == 2:
        b0_2d = b0_map
    else:
        raise ValueError(f"Expected 2D or 4D b0_map, got shape {b0_map.shape}")

    H, W = b0_2d.shape
    device = b0_2d.device

    # Build full basis
    B_full = _build_2d_polynomial_basis((H, W), max_order, device=device)  # [M, K]
    b_full = b0_2d.reshape(-1)  # [M]

    # Apply mask if provided
    if mask is not None:
        mask_2d = mask.squeeze().reshape(-1).bool()
        B_fit = B_full[mask_2d]  # [M_mask, K]
        b_fit = b_full[mask_2d]  # [M_mask]
    else:
        B_fit = B_full
        b_fit = b_full

    # Least-squares solve: c = (B^T B)^{-1} B^T b
    # Use torch.linalg.lstsq for numerical stability
    result = torch.linalg.lstsq(B_fit, b_fit.unsqueeze(1))
    coeffs = result.solution.squeeze(1)  # [K]

    # Evaluate fitted field on the FULL grid (extrapolation)
    fitted_flat = B_full @ coeffs  # [M]
    fitted_map = fitted_flat.reshape(H, W).unsqueeze(0).unsqueeze(0)

    return fitted_map, coeffs


def evaluate_sh_field(
    coefficients: torch.Tensor,
    shape: tuple[int, int],
    max_order: int = 2,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Evaluate a spherical harmonic field from coefficients.

    Args:
        coefficients: SH coefficient vector [K].
        shape: (H, W) output dimensions.
        max_order: Maximum polynomial order matching the coefficients.
        device: Target device.

    Returns:
        Field map [1, 1, H, W] in Hz.
    """
    B = _build_2d_polynomial_basis(shape, max_order, device=device)
    field = B @ coefficients.to(device)
    return field.reshape(shape[0], shape[1]).unsqueeze(0).unsqueeze(0)


__all__ = [
    "evaluate_sh_field",
    "fit_spherical_harmonics",
]
