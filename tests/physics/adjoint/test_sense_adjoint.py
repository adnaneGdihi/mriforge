"""Dot-product adjoint test for sense_forward / sense_adjoint.

Mathematical guarantee
----------------------
For linear operator A (sense_forward) and its adjoint A* (sense_adjoint):

    ⟨A x, y⟩_kspace == ⟨x, A* y⟩_image

must hold to machine precision.  This is the standard dot-product test
(Fessler & Sutton, 2003; Pruessmann et al., SENSE 1999).

Operator definition
-------------------
    sense_forward(img, smaps, mask):
        coil_imgs = img * smaps           # (B, C_coils, H, W) complex
        kspace    = fft2c(coil_imgs)      # (B, C_coils, H, W) complex
        return apply_mask(kspace, mask)

    sense_adjoint(kspace, smaps, mask):
        kspace = apply_mask(kspace, mask)
        img    = ifft2c(kspace)           # (B, C_coils, H, W)
        return Σ_c conj(smaps_c) * img_c  # (B, 1, H, W) complex

Inner products
--------------
    LHS = ⟨A x, y⟩ = Σ_{b,c,h,w} [Ax]_{b,c,h,w} · conj(y_{b,c,h,w})
    RHS = ⟨x, A* y⟩ = Σ_{b,h,w} x_{b,1,h,w} · conj([A*y]_{b,1,h,w})

Design notes
------------
* smaps are built to have unit norm per coil (||smaps[:,c,:,:]||₂ = 1)
  so the numerical scale is controlled.
* The mask is applied symmetrically in both operators — only masked
  k-space positions contribute, which is what makes sense_adjoint the
  true adjoint of sense_forward.
* All computations are CPU, float32-precision (complex64).
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.fft_ops import sense_adjoint, sense_forward
from tests.utils.phantoms import random_complex
from tests.utils.tolerances import tol_for

# ---------------------------------------------------------------------------
# Parametrize axes
# ---------------------------------------------------------------------------

_DTYPES = [torch.complex64]  # fft2c always returns complex64
_SHAPES = [
    (1, 1, 8, 8),
    (2, 1, 16, 16),
    (1, 1, 12, 20),
]
_COILS = [1, 4, 8, 12]


def _mask_all_ones(B: int, H: int, W: int) -> torch.Tensor:
    """All-sampled Cartesian mask, shape (B, 1, H, W)."""
    return torch.ones(B, 1, H, W, dtype=torch.float32)


def _mask_equispaced(B: int, H: int, W: int) -> torch.Tensor:
    """Equispaced every-other-column mask (acceleration ≈ 2)."""
    mask = torch.zeros(B, 1, H, W, dtype=torch.float32)
    mask[:, :, :, ::2] = 1.0
    return mask


def _mask_random(B: int, H: int, W: int, seed: int = 42) -> torch.Tensor:
    """Random binary mask with ~50% sampling fraction."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    bits = torch.randint(0, 2, (B, 1, H, W), generator=gen, dtype=torch.float32)
    # Ensure DC (centre) is always sampled
    bits[:, :, H // 2, W // 2] = 1.0
    return bits


_MASK_FACTORIES = {
    "all_ones": _mask_all_ones,
    "equispaced": _mask_equispaced,
    "random": _mask_random,
}


def _unit_norm_smaps(B: int, C: int, H: int, W: int, seed: int) -> torch.Tensor:
    """Build complex sensitivity maps with unit Frobenius norm per coil.

    Shape: (B, C, H, W) complex64.  Unit-normalising each coil map
    keeps the numerical scale of the inner products bounded and makes
    the adjoint test scale-independent.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    smaps = random_complex((B, C, H, W), dtype=torch.complex64, generator=gen)
    # Normalise per coil: ||smaps[:, c, :, :]||_F = 1 (over spatial dims)
    norms = smaps.norm(dim=(-2, -1), keepdim=True).real.clamp_min(1e-8)
    # norms.shape is (B, C, 1, 1) but .real breaks the complex view; reconstruct:
    norms_c = torch.abs(smaps).norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return smaps / norms_c


def _inner(a: torch.Tensor, b: torch.Tensor) -> complex:
    """⟨a, b⟩ = Σ a_i · conj(b_i) (flattened, complex64)."""
    a = a.to(torch.complex64).flatten()
    b = b.to(torch.complex64).flatten()
    return torch.dot(a, torch.conj(b)).item()


# ---------------------------------------------------------------------------
# The adjoint test proper
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("mask_name", list(_MASK_FACTORIES.keys()))
@pytest.mark.parametrize("n_coils", _COILS, ids=[f"C{c}" for c in _COILS])
@pytest.mark.parametrize("shape", _SHAPES, ids=["x".join(str(d) for d in s) for s in _SHAPES])
def test_sense_dot_product_adjoint(
    shape: tuple[int, ...],
    n_coils: int,
    mask_name: str,
) -> None:
    """⟨sense_forward(x, smaps, mask), y⟩ ≈ ⟨x, sense_adjoint(y, smaps, mask)⟩.

    Tolerance: 100× the complex64 machine tolerance to allow for
    accumulated floating-point errors over N² terms.
    """
    B, C_img, H, W = shape
    seed = hash((shape, n_coils, mask_name)) & 0xFFFF_FFFF

    # Build unit-norm sensitivity maps
    smaps = _unit_norm_smaps(B, n_coils, H, W, seed)

    # Build mask (B, 1, H, W)
    mask = _MASK_FACTORIES[mask_name](B, H, W)

    # x lives in image domain: (B, 1, H, W) complex64
    gen_x = torch.Generator()
    gen_x.manual_seed(seed ^ 0xABCD)
    x = random_complex((B, 1, H, W), dtype=torch.complex64, generator=gen_x)

    # y lives in k-space domain: (B, n_coils, H, W) complex64
    gen_y = torch.Generator()
    gen_y.manual_seed(seed ^ 0x1234)
    y = random_complex((B, n_coils, H, W), dtype=torch.complex64, generator=gen_y)

    # Apply operators
    Ax = sense_forward(x, smaps=smaps, mask=mask)       # (B, n_coils, H, W)
    Astar_y = sense_adjoint(y, smaps=smaps, mask=mask)  # (B, 1, H, W)

    # Inner products
    lhs = _inner(Ax, y)       # ⟨Ax, y⟩_kspace
    rhs = _inner(x, Astar_y)  # ⟨x, A*y⟩_image

    scale = max(abs(lhs), abs(rhs), 1e-8)
    rel_err = abs(lhs - rhs) / scale

    tol = tol_for(torch.complex64) * 100

    assert rel_err < tol, (
        f"SENSE dot-product adjoint FAILED:\n"
        f"  shape={shape}  n_coils={n_coils}  mask={mask_name}\n"
        f"  ⟨Ax, y⟩     = {lhs:.8g}\n"
        f"  ⟨x, A*y⟩    = {rhs:.8g}\n"
        f"  rel_err={rel_err:.2e}  tol={tol:.2e}"
    )


# ---------------------------------------------------------------------------
# Sanity: single-coil (no smaps) sense_forward ≡ fft2c + mask
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.adjoint
@pytest.mark.parametrize("shape", _SHAPES, ids=["x".join(str(d) for d in s) for s in _SHAPES])
def test_sense_no_smaps_is_fft2c(shape: tuple[int, ...]) -> None:
    """sense_forward(x, smaps=None, mask=None) ≡ fft2c(x) (no-smaps path)."""
    from mriforge.infrastructure.physics.fft_ops import fft2c

    B, C, H, W = shape
    gen = torch.Generator()
    gen.manual_seed(hash(shape) & 0xFFFF_FFFF)
    x = random_complex(shape, dtype=torch.complex64, generator=gen)

    via_sense = sense_forward(x, smaps=None, mask=None)
    via_fft = fft2c(x)

    tol = tol_for(torch.complex64) * 10
    assert torch.allclose(via_sense, via_fft, atol=tol, rtol=tol), (
        f"sense_forward(smaps=None) != fft2c for shape={shape}. "
        f"Max err: {(via_sense - via_fft).abs().max().item():.2e}"
    )
