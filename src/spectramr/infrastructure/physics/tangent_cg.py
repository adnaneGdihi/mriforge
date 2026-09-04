r"""Shared conjugate-gradient data-consistency primitive (SOTA plan T1 keystone).

Solves the Tikhonov-regularised normal equations

.. math::
    (A^{H} A + \lambda I)\, x = A^{H} y + \lambda x_0

by conjugate gradient, initialised at :math:`x_0`. This is the single k-space
data-consistency solver consumed by the Decomposed-Diffusion sampler (DDS),
SSDU, Equivariant Imaging, and Ambient/A-DPS, extracted and generalised from the
inline ``DDSSampler._cg_solve``.

Two generalisations over the inline version:

1. It takes arbitrary linear ``forward_op`` (:math:`A`) and ``adjoint_op``
   (:math:`A^{H}`) callables, so the same routine serves identity-, SENSE-, and
   NUFFT-forward models without copy-paste.
2. It uses the **Hermitian** inner product :math:`\langle r, r\rangle =
   \mathrm{Re}\,(r^{H} r)`, so it is correct on COMPLEX k-space. The inline DDS
   solver used ``(r * r).sum()``, which is wrong for complex tensors (it is not
   real-valued and not the squared norm).

Canonical home: the physics layer (next to ``data_consistency.py``), per the
one-directory rule — CG-on-k-space is a physics primitive, not sampler-private.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _real_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hermitian inner product ``Re(<a, b>)`` over all elements.

    Real for real tensors, ``Re(sum(conj(a) * b))`` for complex tensors. This is
    the quantity CG needs (a real, positive curvature measure) regardless of
    whether the data is complex k-space or real image-domain.
    """
    prod = a.conj() * b if torch.is_complex(a) else a * b
    s = prod.sum()
    return s.real if torch.is_complex(s) else s


def cg_data_consistency(
    x0: torch.Tensor,
    forward_op: Callable[[torch.Tensor], torch.Tensor],
    adjoint_op: Callable[[torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    *,
    n_iter: int = 5,
    lam: float = 1e-2,
    tol: float = 1e-6,
) -> torch.Tensor:
    r"""Tikhonov data-consistency solve by conjugate gradient.

    Returns the minimiser of
    :math:`\tfrac12\|A x - y\|_2^2 + \tfrac{\lambda}{2}\|x - x_0\|_2^2`,
    i.e. the solution of :math:`(A^{H}A + \lambda I)x = A^{H}y + \lambda x_0`.

    Args:
        x0: Initial estimate / prior anchor, any shape the operators accept.
        forward_op: Linear operator :math:`A` (e.g. ``MFS`` or identity).
        adjoint_op: Its Hermitian adjoint :math:`A^{H}`.
        y: Measurement in the range of :math:`A`.
        n_iter: Maximum CG iterations (typically 5 for DDS). Must be >= 1.
        lam: Tikhonov weight :math:`\lambda` (anchors the solution to ``x0``).
        tol: Early-stop on residual norm.

    Returns:
        The data-consistent estimate, same shape/dtype as ``x0``.

    Raises:
        ValueError: if ``n_iter < 1`` (a silent no-op would mask a wiring bug).
    """
    if n_iter < 1:
        raise ValueError(f"n_iter must be >= 1, got {n_iter}")

    def normal(v: torch.Tensor) -> torch.Tensor:
        return adjoint_op(forward_op(v)) + lam * v

    rhs = adjoint_op(y) + lam * x0
    x = x0.clone()
    r = rhs - normal(x)
    p = r.clone()
    rs_old = _real_inner(r, r)
    tol2 = tol * tol
    for _ in range(n_iter):
        if rs_old <= tol2:
            break
        Ap = normal(p)
        denom = _real_inner(p, Ap).clamp_min(1e-12)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = _real_inner(r, r)
        p = r + (rs_new / rs_old.clamp_min(1e-12)) * p
        rs_old = rs_new
    return x


__all__ = ["cg_data_consistency"]
