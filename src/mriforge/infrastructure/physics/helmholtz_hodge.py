r"""Helmholtz-Hodge decomposition of motion fields for free-breathing CMR.

On a simply-connected 3D domain :math:`\Omega\subset\mathbb{R}^3` the
Helmholtz-Hodge theorem gives the unique :math:`L^2`-orthogonal splitting

.. math::

    \mathbf{v}(\mathbf{r},t)
        = \nabla\phi(\mathbf{r},t)
        + \nabla\times\mathbf{A}(\mathbf{r},t)
        + \mathbf{h}(\mathbf{r},t),
    \qquad \langle\nabla\phi,\nabla\times\mathbf{A}\rangle_{L^2}=0,

with :math:`\phi` a scalar potential (irrotational), :math:`\mathbf{A}` a
vector potential under the Coulomb gauge :math:`\nabla\!\cdot\mathbf{A}=0`
(solenoidal), and :math:`\mathbf{h}` a harmonic remainder. On a simply
connected domain with the standard boundary conditions
:math:`\mathbf{h}\equiv 0`.

In free-breathing cardiac MRI the observed motion is, to leading order, a
superposition of:

* respiratory translation — *irrotational* (rigid translation along the
  Sup-Inf axis), so :math:`\mathbf{v}_{\rm resp}=\nabla\phi`;
* cardiac contraction — *solenoidal* (incompressible rotation about the
  LV centroid), so :math:`\mathbf{v}_{\rm card}=\nabla\times\mathbf{A}`.

Hodge orthogonality therefore provides a physically grounded inductive
bias for disentangling the two motion components *during* reconstruction.

This module implements the **discrete-exterior-calculus (DEC) Hodge
solver** on a regular 3D grid via the Poisson equation

.. math::

    \Delta\phi = \nabla\!\cdot\mathbf{v} \quad\Rightarrow\quad
    \mathbf{v}_{\rm irrot}=\nabla\phi, \quad \mathbf{v}_{\rm sol}=\mathbf{v}-\mathbf{v}_{\rm irrot}.

The Poisson equation is solved via the periodic-FFT inverse Laplacian.

Exports:

* :func:`divergence` / :func:`curl` — central-difference discrete operators.
* :func:`helmholtz_hodge_decompose` — split :math:`\mathbf{v}` into
  irrotational + solenoidal parts.
* :func:`hodge_orthogonality_residual` — diagnostic
  :math:`|\langle\mathbf{v}_{\rm irrot},\mathbf{v}_{\rm sol}\rangle_{L^2}|`.
"""

from __future__ import annotations

import torch
import torch.fft

__all__ = [
    "curl",
    "divergence",
    "helmholtz_hodge_decompose",
    "hodge_orthogonality_residual",
]


def _pdiff(f: torch.Tensor, axis: int) -> torch.Tensor:
    r"""Periodic central difference :math:`(f_{i+1}-f_{i-1})/2` (circular wrap).

    The Helmholtz solver inverts the Laplacian with a *periodic* FFT, so the
    difference operators feeding/recovering it must also be periodic — i.e.
    true circular convolutions diagonalised by the same FFT. The previous
    implementation used zero-boundary differences (``[..., 1:-1, ...]``), which
    are NOT circular convolutions, so the FFT solve could not invert them and
    the decomposition's ``v_sol`` was not divergence-free (orthogonality broke
    well beyond round-off — ``max|div(v_sol)| ~ O(1)``).
    """
    return 0.5 * (torch.roll(f, -1, dims=axis) - torch.roll(f, 1, dims=axis))


def divergence(v: torch.Tensor) -> torch.Tensor:
    r"""Periodic central-difference divergence :math:`\nabla\!\cdot\mathbf{v}`.

    Args:
        v: ``(..., 3, D, H, W)`` vector field; the component axis is -4 and the
            spatial axes are -3 (z), -2 (y), -1 (x).

    Returns:
        ``(..., D, H, W)`` scalar divergence.
    """
    if v.ndim < 4 or v.shape[-4] != 3:
        raise ValueError(f"v must have shape (..., 3, D, H, W); got {tuple(v.shape)}.")
    vz, vy, vx = v[..., 0, :, :, :], v[..., 1, :, :, :], v[..., 2, :, :, :]
    return _pdiff(vz, -3) + _pdiff(vy, -2) + _pdiff(vx, -1)


def curl(v: torch.Tensor) -> torch.Tensor:
    r"""Periodic central-difference curl :math:`\nabla\times\mathbf{v}`.

    Returns:
        ``(..., 3, D, H, W)`` curl field.
    """
    if v.ndim < 4 or v.shape[-4] != 3:
        raise ValueError(f"v must have shape (..., 3, D, H, W); got {tuple(v.shape)}.")
    vz, vy, vx = v[..., 0, :, :, :], v[..., 1, :, :, :], v[..., 2, :, :, :]
    # Curl_x = ∂vz/∂y − ∂vy/∂z, Curl_y = ∂vx/∂z − ∂vz/∂x, Curl_z = ∂vy/∂x − ∂vx/∂y.
    curl_x = _pdiff(vz, -2) - _pdiff(vy, -3)
    curl_y = _pdiff(vx, -3) - _pdiff(vz, -1)
    curl_z = _pdiff(vy, -1) - _pdiff(vx, -2)
    return torch.stack([curl_x, curl_y, curl_z], dim=-4)


def _periodic_poisson_solve(rhs: torch.Tensor) -> torch.Tensor:
    r"""Solve :math:`\Delta u=\mathrm{rhs}` on a periodic 3D grid via FFT.

    The eigenvalue ``lam`` must be the Fourier symbol of the **same** stencil
    that ``divergence``/``grad`` compose to. Periodic central differences
    applied twice give ``(f_{i+2}-2f_i+f_{i-2})/4`` per axis, whose symbol is
    ``-sin²(2πk)`` — NOT the narrow 3-point ``-4·sin²(πk)`` the previous code
    used. With the matched symbol, ``div(grad(poisson(div v))) = div(v)``
    spectrally, so the recovered ``v_sol`` is divergence-free to round-off.
    """
    d, h, w = rhs.shape[-3:]
    R = torch.fft.fftn(rhs, dim=(-3, -2, -1))
    kz = torch.fft.fftfreq(d, device=rhs.device, dtype=rhs.dtype)
    ky = torch.fft.fftfreq(h, device=rhs.device, dtype=rhs.dtype)
    kx = torch.fft.fftfreq(w, device=rhs.device, dtype=rhs.dtype)
    lam = -(
        torch.sin(2.0 * torch.pi * kz)[:, None, None] ** 2
        + torch.sin(2.0 * torch.pi * ky)[None, :, None] ** 2
        + torch.sin(2.0 * torch.pi * kx)[None, None, :] ** 2
    )
    eps = 1e-12
    safe = torch.where(lam.abs() < eps, torch.full_like(lam, 1.0), lam)
    U = R / safe
    U = torch.where(lam.abs() < eps, torch.zeros_like(U), U)
    return torch.fft.ifftn(U, dim=(-3, -2, -1)).real


def helmholtz_hodge_decompose(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Decompose :math:`\mathbf{v}=\nabla\phi+\mathbf{v}_{\rm sol}` on a periodic grid.

    Args:
        v: ``(3, D, H, W)`` vector field.

    Returns:
        ``(v_irrot, v_sol)`` — the irrotational and solenoidal components.
        Hodge orthogonality :math:`\langle v_{\rm irrot}, v_{\rm sol}\rangle_{L^2}\approx 0`
        holds to discretisation accuracy.
    """
    if v.ndim != 4 or v.shape[0] != 3:
        raise ValueError(f"v must be (3, D, H, W); got {tuple(v.shape)}.")
    div = divergence(v.unsqueeze(0)).squeeze(0)
    phi = _periodic_poisson_solve(div)
    # grad(phi) with the same periodic stencil as the divergence/Poisson symbol.
    grad_z = _pdiff(phi, -3)
    grad_y = _pdiff(phi, -2)
    grad_x = _pdiff(phi, -1)
    v_irrot = torch.stack([grad_z, grad_y, grad_x], dim=0)
    v_sol = v - v_irrot
    return v_irrot, v_sol


def hodge_orthogonality_residual(v_irrot: torch.Tensor, v_sol: torch.Tensor) -> torch.Tensor:
    r"""Diagnostic :math:`|\langle\mathbf{v}_{\rm irrot},\mathbf{v}_{\rm sol}\rangle_{L^2}|`.

    On a simply-connected periodic domain the inner product is zero up to
    discretisation error; a non-zero residual quantifies the boundary /
    discretisation contribution.
    """
    if v_irrot.shape != v_sol.shape:
        raise ValueError("v_irrot and v_sol must have identical shape.")
    return (v_irrot * v_sol).sum().abs()
