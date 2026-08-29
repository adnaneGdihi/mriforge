"""Regression tests for the Connes spectral-triple primitives.

Covers the doc-contract fix on
:func:`mriforge.infrastructure.physics.spectral_triple.intertwining_residual`:
the helper computes a *Dirac-composition* residual
``||U psi - D_HF D_ULF psi||^2`` and **not** the Morita-intertwining defect
``||U D_ULF psi - D_HF U psi||^2`` that the original docstring claimed. The
signature never receives the operator ``U`` (only the tensor ``u_psi = U(psi)``)
so the documented intertwiner is not even computable here. These tests pin the
corrected contract so a future maintainer cannot silently reintroduce the
mismatch.
"""

from __future__ import annotations

import torch

from mriforge.infrastructure.physics.spectral_triple import (
    discrete_dirac_2d,
    intertwining_residual,
)


def test_intertwining_residual_is_composition_not_intertwiner() -> None:
    """``intertwining_residual`` computes ``||U psi - D_HF D_ULF psi||^2``.

    This composition residual is NOT zero even when ``U`` exactly intertwines
    the two Dirac operators (the documented intertwiner *would* vanish), which
    is exactly the doc/code contradiction the fix resolves.
    """
    torch.manual_seed(0)
    n = 6
    U = torch.randn(n, n)
    Dulf = torch.randn(n, n)
    psi = torch.randn(n)
    # Construct D_HF so that U exactly intertwines: U @ Dulf = Dhf @ U.
    Dhf = U @ Dulf @ torch.linalg.inv(U)
    d_ulf = lambda x: Dulf @ x  # noqa: E731
    d_hf = lambda x: Dhf @ x  # noqa: E731
    u_psi = U @ psi

    coded = intertwining_residual(u_psi, psi, d_ulf, d_hf)
    documented = ((U @ d_ulf(psi)) - d_hf(u_psi)).abs().pow(2).mean()

    # The documented Morita intertwiner vanishes under exact intertwining...
    assert torch.allclose(documented, torch.zeros_like(documented), atol=1e-8)
    # ...but the coded composition residual does not.
    assert coded > 1.0
    assert not torch.allclose(coded, documented)


def test_intertwining_residual_matches_explicit_composition_formula() -> None:
    """The returned value equals the mean-squared magnitude of the composition."""
    torch.manual_seed(1)
    psi = torch.randn(5)
    A = torch.randn(5, 5)
    B = torch.randn(5, 5)
    d_ulf = lambda x: A @ x  # noqa: E731
    d_hf = lambda x: B @ x  # noqa: E731
    u_psi = torch.randn(5)

    out = intertwining_residual(u_psi, psi, d_ulf, d_hf)
    expected = (u_psi - d_hf(d_ulf(psi))).abs().pow(2).mean()
    assert torch.allclose(out, expected, atol=1e-8)
    # Scalar output.
    assert out.ndim == 0


def test_intertwining_residual_zero_when_composition_matches() -> None:
    """Residual is exactly zero when ``u_psi == D_HF(D_ULF(psi))``."""
    torch.manual_seed(2)
    psi = torch.randn(4)
    A = torch.randn(4, 4)
    B = torch.randn(4, 4)
    d_ulf = lambda x: A @ x  # noqa: E731
    d_hf = lambda x: B @ x  # noqa: E731
    u_psi = B @ (A @ psi)  # exactly the composition

    out = intertwining_residual(u_psi, psi, d_ulf, d_hf)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-10)


def test_discrete_dirac_2d_smoke_with_helper() -> None:
    """Sanity guard that ``discrete_dirac_2d`` (the helper the loss actually uses)
    runs on a tiny spinor field and preserves shape."""
    torch.manual_seed(3)
    field = torch.randn(1, 2, 4, 4)
    out = discrete_dirac_2d(field, field_strength=1.0)
    assert out.shape == field.shape
    assert torch.is_complex(out)


# ---------------------------------------------------------------------------
# Self-adjointness of the Dirac operator (2026-07 -i fix)
# ---------------------------------------------------------------------------
# discrete_dirac_2d previously omitted the leading -i, so the differential part
# was skew-adjoint, contradicting the spectral-triple requirement of a
# self-adjoint Dirac operator. The -i is a global unit factor, so the
# difference-based spectral_triple_loss is numerically unchanged.


def test_discrete_dirac_is_self_adjoint_on_interior_fields() -> None:
    """<Dψ, φ> == <ψ, Dφ> for spinor fields supported away from the boundary.

    The boundary rows/columns are zeroed (Dirichlet edge), so exact discrete
    self-adjointness holds only for interior-supported fields.
    """
    torch.manual_seed(0)
    H = W = 12
    mask = torch.zeros(1, 1, H, W)
    mask[..., 2:-2, 2:-2] = 1.0

    def interior_spinor() -> torch.Tensor:
        return torch.complex(torch.randn(1, 2, H, W), torch.randn(1, 2, H, W)) * mask

    psi, phi = interior_spinor(), interior_spinor()
    d_psi = discrete_dirac_2d(psi, field_strength=1.3)
    d_phi = discrete_dirac_2d(phi, field_strength=1.3)
    lhs = (d_psi.conj() * phi).sum()
    rhs = (psi.conj() * d_phi).sum()
    assert (lhs - rhs).abs().item() < 1e-4


def test_dirac_difference_loss_is_invariant_to_global_phase() -> None:
    """The -i factor cancels in a difference-based penalty (loss unchanged).

    ||(-i·A) - (-i·B)||² == ||A - B||²; verify the operator's difference
    penalty equals the same penalty computed from the un-scaled σ·∂ action.
    """
    torch.manual_seed(1)
    a = torch.complex(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))
    b = torch.complex(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))
    da = discrete_dirac_2d(a, field_strength=2.0)
    db = discrete_dirac_2d(b, field_strength=2.0)
    with_i = (da - db).abs().pow(2).mean()
    without_i = ((1j * da) - (1j * db)).abs().pow(2).mean()  # remove the -i
    assert torch.allclose(with_i, without_i, atol=1e-6)
