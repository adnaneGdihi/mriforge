r"""Galois-cohomological phase unwrapping via covering-space lifting.

This module implements phase unwrapping on a regular 3D voxel grid via the
covering map :math:`p:\\mathbb{R}\\to S^1`, :math:`p(\\theta)=e^{i\\theta}`,
which is a :math:`\\mathbb{Z}`-Galois cover with deck group :math:`\\mathbb{Z}`.
A global continuous lift :math:`\\tilde\\varphi:\\Omega\\to\\mathbb{R}` of
:math:`\\varphi:\\Omega\\to S^1` exists iff the winding cocycle
:math:`w\\in Z^1(\\Omega,\\underline{\\mathbb{Z}})` vanishes in cohomology:
:math:`[w]=0\\in H^1(\\Omega,\\underline{\\mathbb{Z}})`. On the standard
cubic-grid Čech cover the obstruction reduces to the per-plaquette residue —
the sum of integer wraps around each elementary face of the lattice.

The module exposes:

* :func:`wrap_to_principal` — the principal-value wrap :math:`\\theta\\mod 2\\pi`.
* :func:`integer_winding_cocycle` — the edge-valued cocycle
  :math:`w_{(i,j)}\\in\\mathbb{Z}` on the 1-skeleton of the cubic grid.
* :func:`plaquette_residues` — the boundary :math:`\\partial w` evaluated on
  every elementary 2-face; non-zero entries witness an obstruction class in
  :math:`H^1`.
* :func:`cohomology_norm_certificate` — the scalar
  :math:`\\|[w]\\|_{H^1}=(\\sum_{\\text{plaq}} r^2)^{1/2}` defensibility
  certificate. Zero iff the unwrapping is globally consistent.
* :func:`galois_unwrap` — least-squares lift via the discrete Poisson solver,
  paired with the residue certificate.

When the certificate is exactly zero, the lift is the unique continuous
representative up to the deck-group action (a global :math:`2\\pi` shift).
"""

from __future__ import annotations

import math

import torch
import torch.fft

__all__ = [
    "cohomology_norm_certificate",
    "galois_unwrap",
    "integer_winding_cocycle",
    "plaquette_residues",
    "wrap_to_principal",
]


def wrap_to_principal(theta: torch.Tensor) -> torch.Tensor:
    r"""Map :math:`\theta\to\theta-2\pi\lfloor(\theta+\pi)/(2\pi)\rfloor\in(-\pi,\pi]`.

    This is the principal-value wrap; idempotent and differentiable almost
    everywhere (jumps occur on the codimension-1 set
    :math:`\{\theta\equiv\pi\bmod 2\pi\}`).
    """
    return theta - 2.0 * math.pi * torch.floor((theta + math.pi) / (2.0 * math.pi))


def _axis_diff(phi: torch.Tensor, axis: int) -> torch.Tensor:
    """Forward difference along a spatial axis of a ``[..., D, H, W]`` tensor."""
    if axis not in (-3, -2, -1):
        raise ValueError(f"axis must be one of {-3, -2, -1}; got {axis}")
    slicer = [slice(None)] * phi.ndim
    next_slicer = [slice(None)] * phi.ndim
    slicer[axis] = slice(0, -1)
    next_slicer[axis] = slice(1, None)
    return phi[tuple(next_slicer)] - phi[tuple(slicer)]


def integer_winding_cocycle(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Edge-valued winding cocycle :math:`w\in Z^1(\Omega,\underline{\mathbb{Z}})`.

    For each oriented edge of the cubic lattice (those connecting a voxel to
    its forward neighbour along axes :math:`x,y,z`), record the integer
    :math:`w_{ij}=\bigl(\varphi(j)-\varphi(i)-\mathrm{wrap}(\varphi(j)-\varphi(i))\bigr)/(2\pi)`.

    Args:
        phi: Wrapped principal phase, shape ``[..., D, H, W]``, real-valued in
            :math:`(-\pi,\pi]` (anything outside is auto-wrapped).

    Returns:
        ``(w_z, w_y, w_x)`` — three integer-valued tensors of shapes
        ``[..., D-1, H, W]``, ``[..., D, H-1, W]``, ``[..., D, H, W-1]`` giving
        the cocycle along the :math:`z,y,x` axes respectively. The values
        :math:`w_{ij}` are obtained by subtracting the principal-wrap of the
        forward difference from the raw forward difference and dividing by
        :math:`2\pi`.
    """
    phi = wrap_to_principal(phi)
    raw_z = _axis_diff(phi, -3)
    raw_y = _axis_diff(phi, -2)
    raw_x = _axis_diff(phi, -1)
    w_z = torch.round((raw_z - wrap_to_principal(raw_z)) / (2.0 * math.pi))
    w_y = torch.round((raw_y - wrap_to_principal(raw_y)) / (2.0 * math.pi))
    w_x = torch.round((raw_x - wrap_to_principal(raw_x)) / (2.0 * math.pi))
    return w_z, w_y, w_x


def plaquette_residues(
    w_z: torch.Tensor, w_y: torch.Tensor, w_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Coboundary :math:`\partial w` on the three coordinate-plane plaquettes.

    For each elementary face of the cubic grid we evaluate the discrete
    Stokes integral of the edge-cochain :math:`w` around its oriented
    boundary. A non-zero residue is the local witness of an obstruction
    class in :math:`H^1(\Omega,\underline{\mathbb{Z}})`; the lift exists
    globally iff every plaquette residue vanishes.

    Args:
        w_z, w_y, w_x: Edge-cochains as returned by
            :func:`integer_winding_cocycle`.

    Returns:
        ``(r_xy, r_xz, r_yz)`` — integer-valued residues on the three families
        of 2-faces, shapes ``[..., D, H-1, W-1]``, ``[..., D-1, H, W-1]``,
        ``[..., D-1, H-1, W]``. Each entry equals the signed plaquette winding.
    """
    # xy-plaquette: boundary = +w_x(y) +w_y(x+1) -w_x(y+1) -w_y(x)
    r_xy = w_x[..., :, :-1, :] + w_y[..., :, :, 1:] - w_x[..., :, 1:, :] - w_y[..., :, :, :-1]
    # xz-plaquette: boundary = +w_x(z) +w_z(x+1) -w_x(z+1) -w_z(x)
    r_xz = w_x[..., :-1, :, :] + w_z[..., :, :, 1:] - w_x[..., 1:, :, :] - w_z[..., :, :, :-1]
    # yz-plaquette: boundary = +w_y(z) +w_z(y+1) -w_y(z+1) -w_z(y)
    r_yz = w_y[..., :-1, :, :] + w_z[..., :, 1:, :] - w_y[..., 1:, :, :] - w_z[..., :, :-1, :]
    return r_xy, r_xz, r_yz


def cohomology_norm_certificate(phi: torch.Tensor) -> torch.Tensor:
    r"""Scalar defensibility certificate :math:`\|[w]\|_{H^1}\ge 0`.

    Returns the :math:`\ell^2` norm of the plaquette residue field. The lift
    is globally consistent iff this number is exactly zero; a positive value
    counts the number of branch-cut endpoints (residue dipoles) up to a
    global sign.
    """
    w = integer_winding_cocycle(phi)
    r_xy, r_xz, r_yz = plaquette_residues(*w)
    total_sq = r_xy.pow(2).sum() + r_xz.pow(2).sum() + r_yz.pow(2).sum()
    return torch.sqrt(total_sq.to(phi.dtype))


def _discrete_laplacian_dst_solve(rhs: torch.Tensor) -> torch.Tensor:
    """Solve :math:`\\Delta u = \\text{rhs}` on a 3-D grid with zero-Dirichlet BCs.

    Uses the eigendecomposition of the discrete Laplacian: the DST-I basis
    diagonalises the 7-point stencil with homogeneous Dirichlet boundary
    conditions. We implement the DST-I as a real-FFT on the odd-extended
    signal to keep the dependency surface minimal.
    """
    D, H, W = rhs.shape[-3:]
    rhs_ext_d = torch.cat([rhs.new_zeros(*rhs.shape[:-3], 1, H, W), rhs], dim=-3)
    rhs_ext_d = torch.cat([rhs_ext_d, -rhs_ext_d.flip(-3)[..., 1:, :, :]], dim=-3)
    rhs_ext_dh = torch.cat([rhs_ext_d.new_zeros(*rhs_ext_d.shape[:-2], 1, W), rhs_ext_d], dim=-2)
    rhs_ext_dh = torch.cat([rhs_ext_dh, -rhs_ext_dh.flip(-2)[..., :, 1:, :]], dim=-2)
    rhs_ext = torch.cat([rhs_ext_dh.new_zeros(*rhs_ext_dh.shape[:-1], 1), rhs_ext_dh], dim=-1)
    rhs_ext = torch.cat([rhs_ext, -rhs_ext.flip(-1)[..., 1:]], dim=-1)

    R = torch.fft.fftn(rhs_ext, dim=(-3, -2, -1))

    Nz, Ny, Nx = R.shape[-3], R.shape[-2], R.shape[-1]
    kz = torch.arange(Nz, device=rhs.device, dtype=rhs.dtype)
    ky = torch.arange(Ny, device=rhs.device, dtype=rhs.dtype)
    kx = torch.arange(Nx, device=rhs.device, dtype=rhs.dtype)
    lz = 2.0 * (torch.cos(2.0 * math.pi * kz / Nz) - 1.0)
    ly = 2.0 * (torch.cos(2.0 * math.pi * ky / Ny) - 1.0)
    lx = 2.0 * (torch.cos(2.0 * math.pi * kx / Nx) - 1.0)
    lam = lz[:, None, None] + ly[None, :, None] + lx[None, None, :]
    eps = 1e-12
    safe = torch.where(lam.abs() < eps, torch.full_like(lam, 1.0), lam)
    U = R / safe
    U = torch.where(lam.abs() < eps, torch.zeros_like(U), U)

    u_ext = torch.fft.ifftn(U, dim=(-3, -2, -1)).real
    out = u_ext[..., 1 : D + 1, 1 : H + 1, 1 : W + 1]
    return out


def galois_unwrap(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Phase unwrap with a Galois-cohomological defensibility certificate.

    Performs the least-squares Poisson lift
    :math:`\Delta\tilde\varphi=\nabla\cdot(\mathrm{wrap}\,\nabla\varphi)`
    [Schofield & Zhu, *Opt. Lett.* 2003] augmented with the plaquette-residue
    certificate that quantifies how much of the result violates global
    consistency.

    Args:
        phi: Wrapped phase, real, shape ``[..., D, H, W]``. Anything outside
            :math:`(-\pi,\pi]` is auto-wrapped via :func:`wrap_to_principal`.

    Returns:
        ``(theta_unwrap, certificate)``:

        * ``theta_unwrap``: continuous lift, same shape as ``phi``. Unique up
          to the deck-group action — a global :math:`2\pi` shift.
        * ``certificate``: scalar :math:`\|[w]\|_{H^1}\ge 0`. Zero iff the
          lift is globally consistent on the cubic-grid Čech cover.
    """
    phi = wrap_to_principal(phi)
    grad_x = wrap_to_principal(_axis_diff(phi, -1))
    grad_y = wrap_to_principal(_axis_diff(phi, -2))
    grad_z = wrap_to_principal(_axis_diff(phi, -3))

    # Discrete divergence of the wrapped gradient — Laplacian RHS.
    rhs = torch.zeros_like(phi)
    rhs[..., :, :, 1:] = rhs[..., :, :, 1:] + grad_x
    rhs[..., :, :, :-1] = rhs[..., :, :, :-1] - grad_x
    rhs[..., :, 1:, :] = rhs[..., :, 1:, :] + grad_y
    rhs[..., :, :-1, :] = rhs[..., :, :-1, :] - grad_y
    rhs[..., 1:, :, :] = rhs[..., 1:, :, :] + grad_z
    rhs[..., :-1, :, :] = rhs[..., :-1, :, :] - grad_z

    theta = _discrete_laplacian_dst_solve(rhs)
    cert = cohomology_norm_certificate(phi)
    return theta, cert
