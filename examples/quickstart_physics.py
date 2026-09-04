"""Quickstart: verify FFT round-trip and adjoint identity in the physics SSOT.

This script demonstrates two correctness invariants of the MRI physics layer:

1. ``ifft2c(fft2c(x)) == x`` to floating-point tolerance.
2. The adjoint identity for the undersampled Fourier **encoding** operator
   ``A = M . fft2c``, whose adjoint is ``A^H = ifft2c . M``:
   ``<A x, y> == <x, A^H y>`` in the complex inner product.

   The second one is the load-bearing check, and the obvious cheaper spelling of
   it is worthless. ``<M x, y> == <x, M y>`` for a real diagonal ``M`` is just
   commutativity of an elementwise product: measured 2026-08-28, it returns a gap
   of **exactly 0.0** for the real mask, for an all-zero mask, and for a random
   dense tensor that is not a mask at all. It cannot fail, so it establishes
   nothing. Routing the identity through ``fft2c``/``ifft2c`` makes it a real
   test of the physics SSOT -- it is ``norm="ortho"`` that makes ``ifft2c`` the
   adjoint of ``fft2c`` rather than a scaled version of it. Planted against this
   script the same day, relative gap correct = 1.4e-07 vs: raw ``torch.fft.fft2``
   (norm="backward", the CLAUDE.md non-negotiable 2 failure mode) = 9.8e-01;
   adjoint dropping the mask = 9.9e-01; adjoint using ``fft2c`` twice = 9.8e-01.

Designed to complete in under 30 seconds on CPU.

Run:

    python examples/quickstart_physics.py
"""

from __future__ import annotations

import torch

import spectramr  # noqa: F401
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c


def _fft_round_trip(side: int = 64) -> float:
    """Verify ifft2c(fft2c(x)) == x for a complex tensor.

    The physics SSOT's fft2c expects a complex-typed tensor (..., H, W) with
    DC at center; the "real/imag as channels" view is interleaved by other
    helpers, not by fft2c itself.
    """
    torch.manual_seed(0)
    x = torch.randn(1, side, side, dtype=torch.complex64)
    x_back = ifft2c(fft2c(x))
    return (x - x_back).abs().max().item()


def _adjoint_identity(side: int = 64) -> tuple[complex, complex, float]:
    """Verify <A x, y> == <x, A^H y> for A = M . fft2c, A^H = ifft2c . M.

    Returns the two inner products and the gap **relative** to their magnitude;
    an absolute gap is not comparable across image sizes, because the inner
    product grows with the number of voxels.
    """
    torch.manual_seed(0)
    x = torch.randn(1, side, side, dtype=torch.complex64)
    y = torch.randn(1, side, side, dtype=torch.complex64)
    # Cartesian undersampling: keep every other phase-encode column.
    mask = torch.zeros(side, side)
    mask[:, ::2] = 1.0

    def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Complex inner product <a, b> = sum(conj(a) * b)."""
        return torch.sum(a.conj() * b)

    lhs = inner(mask * fft2c(x), y)  # <A x, y>
    rhs = inner(x, ifft2c(mask * y))  # <x, A^H y>
    rel_gap = (lhs - rhs).abs().item() / max(lhs.abs().item(), 1e-12)
    return complex(lhs), complex(rhs), rel_gap


def main() -> None:
    rt_err = _fft_round_trip()
    print(f"FFT round-trip max abs error: {rt_err:.2e}")
    if rt_err > 1e-5:
        raise SystemExit("FFT round-trip exceeds tolerance — physics SSOT regression?")
    lhs, rhs, rel_gap = _adjoint_identity()
    print(f"Adjoint identity:  <A x, y>   = {lhs.real:.6f}{lhs.imag:+.6f}j")
    print(f"                   <x, A^H y> = {rhs.real:.6f}{rhs.imag:+.6f}j")
    print(f"                   rel gap    = {rel_gap:.2e}")
    if rel_gap > 1e-5:
        raise SystemExit(
            "Adjoint identity violated for A = M . fft2c. Either fft2c lost its "
            'norm="ortho" (so ifft2c is no longer its adjoint) or the adjoint '
            "no longer applies the same mask."
        )
    print("OK")


if __name__ == "__main__":
    main()
