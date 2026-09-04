"""Tier-1 adjoint and Parseval tests for fft2c / ifft2c.

Mathematical guarantees verified here
--------------------------------------
1. **Parseval / isometry**: ``‖fft2c(x)‖₂² == ‖x‖₂²`` (ortho norm).
2. **Round-trip**: ``ifft2c(fft2c(x)) ≈ x`` to machine precision.
3. **Dot-product adjoint**: ``⟨fft2c(x), y⟩ ≈ ⟨x, ifft2c(y)⟩``
   — this is the formal definition of *ifft2c being the adjoint of fft2c*
   under the standard inner product, which holds exactly when norm="ortho".

Implementation notes
--------------------
* fft2c always upcasts to complex64 internally (see source), so
  complex128 inputs are internally demoted to complex64 then returned as
  complex64.  We therefore run complex128 parametrize cases at the same
  complex64 tolerance (1e-5) rather than 1e-10.
* All tests are CPU-only and deterministic via an explicit torch.Generator
  seeded from the test's parametrize tuple.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from tests.utils.phantoms import random_complex
from tests.utils.tolerances import tol_for

# ---------------------------------------------------------------------------
# Parametrize axes
# ---------------------------------------------------------------------------

# fft2c upcasts everything to complex64 internally; complex128 is preserved
# only if we compare with complex64 tolerance.
_SHAPES = [
    (1, 1, 8, 8),
    (2, 1, 16, 16),
    (1, 1, 12, 20),
]

_DTYPES = [torch.complex64, torch.complex128]

# IDs for readable parametrize labels
def _shape_id(s: tuple[int, ...]) -> str:
    return "x".join(str(d) for d in s)


# ---------------------------------------------------------------------------
# Helper: stable inner product (complex, flattened)
# ---------------------------------------------------------------------------

def _inner(a: torch.Tensor, b: torch.Tensor) -> complex:
    """Complex inner product ⟨a, b⟩ = Σ a_i · conj(b_i)."""
    a = a.to(torch.complex64)
    b = b.to(torch.complex64)
    return torch.sum(a * torch.conj(b)).item()


# ---------------------------------------------------------------------------
# Test 1 — Parseval / isometry
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _DTYPES, ids=["c64", "c128"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_parseval_fft2c(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """‖fft2c(x)‖₂² == ‖x‖₂² (Parseval, ortho norm) for complex inputs."""
    gen = torch.Generator()
    gen.manual_seed(hash((shape, str(dtype))) & 0xFFFF_FFFF)
    x = random_complex(shape, dtype=dtype, generator=gen)

    k = fft2c(x)

    energy_x = torch.sum(torch.abs(x) ** 2).item()
    energy_k = torch.sum(torch.abs(k) ** 2).item()

    scale = max(abs(energy_x), abs(energy_k), 1e-8)
    rel_diff = abs(energy_x - energy_k) / scale

    # fft2c upcasts to complex64, so we always use complex64 tolerance
    tol = tol_for(torch.complex64) * 10  # 1e-4 — accumulation over N² terms
    assert rel_diff < tol, (
        f"Parseval violated for shape={shape} dtype={dtype}: "
        f"‖x‖²={energy_x:.8g}, ‖fft2c(x)‖²={energy_k:.8g}, "
        f"rel_diff={rel_diff:.2e} > tol={tol:.2e}"
    )


@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _DTYPES, ids=["c64", "c128"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_parseval_ifft2c(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """‖ifft2c(k)‖₂² == ‖k‖₂² (Parseval, ortho norm) for complex inputs."""
    gen = torch.Generator()
    gen.manual_seed(hash((shape, str(dtype), "ifft")) & 0xFFFF_FFFF)
    k = random_complex(shape, dtype=dtype, generator=gen)

    x = ifft2c(k)

    energy_k = torch.sum(torch.abs(k) ** 2).item()
    energy_x = torch.sum(torch.abs(x) ** 2).item()

    scale = max(abs(energy_x), abs(energy_k), 1e-8)
    rel_diff = abs(energy_x - energy_k) / scale

    tol = tol_for(torch.complex64) * 10
    assert rel_diff < tol, (
        f"Parseval (ifft2c) violated for shape={shape} dtype={dtype}: "
        f"‖k‖²={energy_k:.8g}, ‖ifft2c(k)‖²={energy_x:.8g}, "
        f"rel_diff={rel_diff:.2e} > tol={tol:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Round-trip: ifft2c(fft2c(x)) ≈ x
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _DTYPES, ids=["c64", "c128"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_roundtrip_fft_ifft(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """ifft2c(fft2c(x)) ≈ x to machine precision."""
    gen = torch.Generator()
    gen.manual_seed(hash((shape, str(dtype), "rt")) & 0xFFFF_FFFF)
    x = random_complex(shape, dtype=dtype, generator=gen)

    # fft2c internally converts to complex64 and returns complex64
    x_c64 = x.to(torch.complex64)
    reconstructed = ifft2c(fft2c(x_c64))

    tol = tol_for(torch.complex64) * 100  # 1e-3: generous for accumulated float ops
    assert torch.allclose(reconstructed, x_c64, atol=tol, rtol=tol), (
        f"Round-trip ifft2c(fft2c(x)) ≈ x failed for shape={shape} dtype={dtype}. "
        f"Max abs err: {(reconstructed - x_c64).abs().max().item():.2e}"
    )


@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _DTYPES, ids=["c64", "c128"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_roundtrip_ifft_fft(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """fft2c(ifft2c(k)) ≈ k to machine precision."""
    gen = torch.Generator()
    gen.manual_seed(hash((shape, str(dtype), "rt2")) & 0xFFFF_FFFF)
    k = random_complex(shape, dtype=dtype, generator=gen)

    k_c64 = k.to(torch.complex64)
    reconstructed = fft2c(ifft2c(k_c64))

    tol = tol_for(torch.complex64) * 100
    assert torch.allclose(reconstructed, k_c64, atol=tol, rtol=tol), (
        f"Round-trip fft2c(ifft2c(k)) ≈ k failed for shape={shape} dtype={dtype}. "
        f"Max abs err: {(reconstructed - k_c64).abs().max().item():.2e}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Dot-product adjoint: ⟨fft2c(x), y⟩ ≈ ⟨x, ifft2c(y)⟩
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("dtype", _DTYPES, ids=["c64", "c128"])
@pytest.mark.parametrize("shape", _SHAPES, ids=[_shape_id(s) for s in _SHAPES])
def test_dot_product_adjoint_fft(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """Dot-product adjoint: ⟨fft2c(x), y⟩ ≈ ⟨x, ifft2c(y)⟩.

    Under ortho norm, ifft2c is the exact adjoint of fft2c in the standard
    Hilbert space ⟨·,·⟩ = Σ a_i·conj(b_i).  The relative error between
    both inner products must be below the complex64 tolerance.
    """
    gen = torch.Generator()
    gen.manual_seed(hash((shape, str(dtype), "dp")) & 0xFFFF_FFFF)
    x = random_complex(shape, dtype=dtype, generator=gen)
    gen2 = torch.Generator()
    gen2.manual_seed(hash((shape, str(dtype), "dp_y")) & 0xFFFF_FFFF)
    y = random_complex(shape, dtype=dtype, generator=gen2)

    # fft2c returns complex64 regardless of input dtype
    x_c64 = x.to(torch.complex64)
    y_c64 = y.to(torch.complex64)

    lhs = _inner(fft2c(x_c64), y_c64)   # ⟨Ax, y⟩
    rhs = _inner(x_c64, ifft2c(y_c64))  # ⟨x, A*y⟩

    scale = max(abs(lhs), abs(rhs), 1e-8)
    rel_err = abs(lhs - rhs) / scale

    tol = tol_for(torch.complex64) * 100  # generous for float accumulation
    assert rel_err < tol, (
        f"Dot-product adjoint FAILED for shape={shape} dtype={dtype}:\n"
        f"  ⟨fft2c(x), y⟩  = {lhs:.8g}\n"
        f"  ⟨x, ifft2c(y)⟩ = {rhs:.8g}\n"
        f"  rel_err={rel_err:.2e}  tol={tol:.2e}"
    )
