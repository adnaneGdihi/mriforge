r"""Connes spectral-triple primitives for MRI cross-field translation.

A *spectral triple* :math:`(\mathcal{A},\mathcal{H},D)` consists of a
:math:`\ast`-algebra :math:`\mathcal{A}` represented on a Hilbert space
:math:`\mathcal{H}` together with a self-adjoint Dirac operator :math:`D`
whose resolvent :math:`(D-z)^{-1}` is compact and whose commutators
:math:`[D,a]` are bounded for all :math:`a\in\mathcal{A}`
[Connes 1994; Connes-Marcolli 2008].

For an MRI image grid :math:`\Omega\subset\mathbb{Z}^2` we take

* :math:`\mathcal{H}=L^2(\Omega,\mathbb{C}^{2})` — two-component spinor
  field over the grid;
* :math:`\mathcal{A}` — pointwise-multiplication operators (i.e. images);
* :math:`D_{B_0}=\gamma B_0\,\sigma\cdot\nabla + V_{B_0}` — a discrete
  Pauli-Dirac operator with potential :math:`V_{B_0}` carrying the
  field-strength dependence.

ULF→HF translation is then a *Morita morphism* of spectral triples — a
unitary intertwiner :math:`U` whose conjugate maps :math:`D_{B_0^{\rm ULF}}`
to :math:`D_{B_0^{\rm HF}}+\Delta D`. Connes' distance formula

.. math::

    d(\varphi,\psi) = \sup_{a\in\mathcal{A}}\{\,|\varphi(a)-\psi(a)| : \|[D,a]\|\le 1\}

provides an intrinsic task-relevant metric on state pairs.

This module exposes the bare mathematical operators; the
field-translation training loss is registered as
:class:`mriforge.models.losses.spectral_triple_loss.SpectralTripleIntertwiningLoss`.
"""

from __future__ import annotations

import torch

__all__ = [
    "connes_distance_upper_bound",
    "dirac_commutator",
    "discrete_dirac_2d",
    "intertwining_residual",
]


def discrete_dirac_2d(
    field: torch.Tensor,
    *,
    field_strength: float = 1.0,
    potential: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Discrete Pauli-Dirac operator on a 2D grid.

    Implements :math:`D_{B_0}\psi=-i\,\gamma B_0\bigl(\sigma_x\partial_x+\sigma_y\partial_y\bigr)\psi+V\psi`,
    with :math:`\sigma_x,\sigma_y` the Pauli matrices acting on the spinor
    index. The leading :math:`-i` makes the differential part self-adjoint
    (:math:`\partial` is skew-adjoint, :math:`\sigma` Hermitian, so
    :math:`(-i\sigma\partial)^\dagger=-i\sigma\partial`) — the defining property
    of the Dirac operator in a spectral triple. It is a global unit-modulus
    factor, so difference-based losses built on this operator are unchanged.

    The two spinor components are stored as the second-to-last channel pair:
    ``field`` has shape ``[..., 2, H, W]``. Centered finite differences are used
    in the interior; the boundary rows/columns are set to zero (Dirichlet-like
    edge), so exact discrete self-adjointness holds for fields supported away
    from the boundary.

    Args:
        field: Spinor field of shape ``[..., 2, H, W]``. Real or complex.
        field_strength: Scalar :math:`B_0` factor (1.0 by default).
        potential: Optional pointwise multiplication potential of shape
            ``[..., H, W]`` or broadcastable; defaults to zero.

    Returns:
        Tensor of the same shape as ``field`` containing :math:`D_{B_0}\psi`.
    """
    if field.shape[-3] != 2:
        raise ValueError(f"field must have 2 spinor components on axis -3; got {field.shape[-3]}.")
    psi_up = field[..., 0, :, :]
    psi_dn = field[..., 1, :, :]
    grad_x_up = torch.zeros_like(psi_up)
    grad_x_dn = torch.zeros_like(psi_dn)
    grad_y_up = torch.zeros_like(psi_up)
    grad_y_dn = torch.zeros_like(psi_dn)
    grad_x_up[..., :, 1:-1] = 0.5 * (psi_up[..., :, 2:] - psi_up[..., :, :-2])
    grad_x_dn[..., :, 1:-1] = 0.5 * (psi_dn[..., :, 2:] - psi_dn[..., :, :-2])
    grad_y_up[..., 1:-1, :] = 0.5 * (psi_up[..., 2:, :] - psi_up[..., :-2, :])
    grad_y_dn[..., 1:-1, :] = 0.5 * (psi_dn[..., 2:, :] - psi_dn[..., :-2, :])
    # Pauli σ_x: swap components. Pauli σ_y: ±i times swap.
    sigma_x_up = grad_x_dn
    sigma_x_dn = grad_x_up
    sigma_y_up = -1j * grad_y_dn
    sigma_y_dn = 1j * grad_y_up
    if not field.is_complex():
        complex_field = torch.complex(field, torch.zeros_like(field))
    else:
        complex_field = field
    # Leading -i: D = -i·γB0·(σ_x ∂_x + σ_y ∂_y) + V. Without it the differential
    # part is skew-adjoint, not self-adjoint. It is a global unit factor, so the
    # difference-based spectral_triple_loss is numerically unchanged.
    out_up = field_strength * (-1j) * (sigma_x_up + sigma_y_up)
    out_dn = field_strength * (-1j) * (sigma_x_dn + sigma_y_dn)
    if potential is not None:
        if potential.is_complex():
            v = potential
        else:
            v = torch.complex(potential, torch.zeros_like(potential))
        out_up = out_up + v * complex_field[..., 0, :, :]
        out_dn = out_dn + v * complex_field[..., 1, :, :]
    return torch.stack([out_up, out_dn], dim=-3)


def dirac_commutator(
    dirac_apply: callable,
    multiplier: torch.Tensor,
    field: torch.Tensor,
) -> torch.Tensor:
    r"""Compute :math:`[D, M_a]\psi=D(a\psi)-a(D\psi)` numerically.

    ``dirac_apply`` is a callable taking ``field`` and returning :math:`D\psi`.
    ``multiplier`` has shape ``[..., H, W]`` and acts by pointwise
    multiplication on each spinor component.
    """
    if multiplier.is_complex():
        a = multiplier
    else:
        a = torch.complex(multiplier, torch.zeros_like(multiplier))
    field_c = field if field.is_complex() else torch.complex(field, torch.zeros_like(field))
    a_expanded = a.unsqueeze(-3)
    return dirac_apply(a_expanded * field_c) - a_expanded * dirac_apply(field_c)


def intertwining_residual(
    u_psi: torch.Tensor,
    psi: torch.Tensor,
    d_ulf: callable,
    d_hf: callable,
) -> torch.Tensor:
    r"""Dirac-composition residual :math:`\|U\psi - D_{\rm HF}D_{\rm ULF}\psi\|^2`.

    Given the translated state :math:`U\psi` (passed as ``u_psi``) and the two
    Dirac operators applied as callables, returns the mean squared magnitude of
    :math:`U\psi - D_{\rm HF}(D_{\rm ULF}(\psi))`.

    .. note::

        This is a *composition* residual, **not** the Morita-intertwining
        defect :math:`\|U D_{\rm ULF}\psi - D_{\rm HF}U\psi\|^2`: the signature
        receives ``u_psi = U(\psi)`` (a tensor), not the operator :math:`U`, so
        the intertwiner :math:`U(D_{\rm ULF}(\psi))` cannot be evaluated here.
        The registered training loss
        :class:`~mriforge.models.losses.spectral_triple_loss.SpectralTripleIntertwiningLoss`
        does **not** call this helper; it computes its own inline Dirac penalty
        from :func:`discrete_dirac_2d`. To compute the true intertwiner, pass the
        operator :math:`U` as a callable and return
        ``(u_apply(d_ulf(psi)) - d_hf(u_psi)).abs().pow(2).mean()`` instead.
    """
    return (u_psi - d_hf.__call__(d_ulf(psi))).abs().pow(2).mean()


def connes_distance_upper_bound(state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
    r"""Cauchy-Schwarz upper bound on the Connes distance between two states.

    The exact Connes distance involves an SDP over all
    :math:`\|[D,a]\|\le 1`; we bound it via Cauchy-Schwarz applied to the
    :math:`L^2` norm:

    .. math::

        d_{\rm Connes}(\varphi,\psi) \le \|\varphi-\psi\|_2.

    The bound is sharp on rank-1 spectral triples and useful as a
    conservative regulariser in training.
    """
    return (state_a - state_b).reshape(-1).norm(p=2)
