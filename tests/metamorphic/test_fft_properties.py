"""Metamorphic tests for FFT physical properties.

Metamorphic relations encode how the output changes when the input is
transformed in a known way.  If the software violates these relations,
either the implementation is wrong or it does not satisfy basic signal-
processing requirements.

Properties verified
-------------------
1. **Hermitian symmetry**: fft2c of a *real-valued* input has a conjugate-
   symmetric spectrum:  K[-k] = conj(K[k]).  (Discrete form: roll by
   (H//2, W//2) and check K == conj(flip(K)) up to the DC row/col.)

2. **Translation → linear phase shift in k-space**: shifting the image by
   integer pixels (p, q) multiplies each k-space frequency (ky, kx) by
   ``exp(-i 2π (ky·p/H + kx·q/W))``.  We verify numerically that
   ``fft2c(roll(x, p, q)) ≈ fft2c(x) * phase_ramp(p, q)``.

3. **Shift-invariance of energy**: circular shifts preserve the energy
   ``‖fft2c(roll(x))‖₂² == ‖fft2c(x)‖₂²``.

All tests are CPU-only and deterministic via seeded generators.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from tests.utils.phantoms import random_complex
from tests.utils.tolerances import tol_for

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_real(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Random float32 tensor (real-valued), which we'll convert to complex64."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    real = torch.randn(shape, dtype=torch.float32, generator=gen)
    return real.to(torch.complex64)  # zero imaginary part


def _phase_ramp(p: int, q: int, H: int, W: int) -> torch.Tensor:
    """Compute the frequency-domain phase ramp for a (p, q) pixel shift.

    For a circular shift by (p rows, q cols), the DFT relationship is:
        DFT{x[n-p]}[k] = DFT{x[n]}[k] * exp(-i 2π k p / N)

    We build the 2D version as an outer product of 1D ramps.
    ``fft2c`` returns CENTERED k-space -- DC at the middle, not at index 0 --
    so the ramp has to be evaluated on the centered frequency grid too. The
    previous version built it from bare ``fftfreq`` under a docstring claiming
    that matched "fft2c's output layout"; it is the opposite layout, and the
    comment beside it ("already uncentered") named the error while drawing the
    other conclusion. Measured on a 16x16 complex input, shift=(1, 0):

        bare fftfreq          max|K_shifted - K*ramp| = 4.15e+00
        fftshift(fftfreq)     max|K_shifted - K*ramp| = 3.63e-07

    i.e. `fft2c` obeys the shift theorem to float32 noise, and the test was
    asserting it against the wrong convention.

    Returns: complex64 tensor (1, 1, H, W) phase ramp.
    """
    ky = torch.fft.fftshift(torch.fft.fftfreq(H, d=1.0))
    kx = torch.fft.fftshift(torch.fft.fftfreq(W, d=1.0))
    # Phase factors: exp(-i 2π ky p) and exp(-i 2π kx q)
    ky_phase = torch.exp(-1j * 2 * math.pi * ky * p).to(torch.complex64)
    kx_phase = torch.exp(-1j * 2 * math.pi * kx * q).to(torch.complex64)
    # 2D ramp: outer product — shape (H, W)
    ramp_2d = torch.outer(ky_phase, kx_phase)
    return ramp_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


# ---------------------------------------------------------------------------
# 1. Hermitian symmetry for real-valued input
# ---------------------------------------------------------------------------

@pytest.mark.metamorphic
@pytest.mark.physics
@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 1, 16, 16), (1, 1, 12, 20)])
def test_hermitian_symmetry_real_input(shape: tuple[int, ...]) -> None:
    """fft2c of a real image is conjugate-symmetric: K[k] = conj(K[-k]).

    We verify this numerically using the discrete relation:
        K[ky, kx] ≈ conj(K[-ky, -kx])
    where indices wrap around (circular).

    Note: The property holds for torch.fft.fft2 exactly; here we verify
    it also holds for fft2c (which applies fftshift/ifftshift).
    """
    B, C, H, W = shape
    seed = hash(("herm_sym", shape)) & 0xFFFF_FFFF
    x = _random_real(shape, seed)  # real-valued image as complex with zero imag

    K = fft2c(x)  # complex64

    # Conjugate-symmetric check: K[ky, kx] == conj(K[-ky % H, -kx % W])
    # torch.roll with negative index is the same as (-n) % N
    K_flipped = torch.flip(K, dims=[-2, -1])  # flip both spatial dims = negate freq
    # After flip, index (0, 0) becomes (H-1, W-1) unless we account for DC.
    # The correct Hermitian relation on the DFT grid is:
    #   K[k] = conj(K[N - k])  where k=0 maps to k=0 (DC is self-conjugate)
    # torch.flip reverses [0..N-1] → [N-1..0], not [N, N-1..1, 0],
    # so we need to roll by 1 after flipping to get the right pairing.
    K_conj_sym = torch.roll(K_flipped.conj(), shifts=(1, 1), dims=(-2, -1))

    # For purely real input, K and K_conj_sym should agree
    tol = tol_for(torch.complex64) * 100
    max_err = (K - K_conj_sym).abs().max().item()
    assert max_err < tol, (
        f"Hermitian symmetry violated for shape={shape}. "
        f"max|K[k] - conj(K[-k])| = {max_err:.2e}  tol={tol:.2e}"
    )


# ---------------------------------------------------------------------------
# 2. Translation → linear phase ramp in k-space
# ---------------------------------------------------------------------------

@pytest.mark.metamorphic
@pytest.mark.physics
@pytest.mark.parametrize("shift", [(0, 1), (1, 0), (2, 3), (4, 4)])
@pytest.mark.parametrize("shape", [(1, 1, 16, 16), (2, 1, 12, 20)])
def test_translation_phase_ramp(
    shape: tuple[int, ...],
    shift: tuple[int, int],
) -> None:
    """Circular shift of image == multiplication by phase ramp in k-space.

    Metamorphic relation:
        fft2c(roll(x, p, q)) == fft2c(x) * phase_ramp(p, q)

    Here 'roll' is circular shift: ``torch.roll(x, (p, q), dims=(-2, -1))``.
    """
    B, C, H, W = shape
    p, q = shift
    seed = hash(("trans_phase", shape, shift)) & 0xFFFF_FFFF

    gen = torch.Generator()
    gen.manual_seed(seed)
    x = random_complex(shape, dtype=torch.complex64, generator=gen)

    # Shifted image
    x_shifted = torch.roll(x, shifts=(p, q), dims=(-2, -1))

    # FFT of both
    K = fft2c(x)
    K_shifted = fft2c(x_shifted)

    # Expected: K_shifted ≈ K * phase_ramp
    ramp = _phase_ramp(p, q, H, W)  # (1, 1, H, W)
    K_expected = K * ramp

    tol = tol_for(torch.complex64) * 100
    max_err = (K_shifted - K_expected).abs().max().item()
    assert max_err < tol, (
        f"Translation phase ramp violated for shape={shape}, shift={shift}.\n"
        f"  max|K_shifted - K*ramp| = {max_err:.2e}  tol={tol:.2e}"
    )


# ---------------------------------------------------------------------------
# 3. Shift invariance of energy (Parseval + translation)
# ---------------------------------------------------------------------------

@pytest.mark.metamorphic
@pytest.mark.physics
@pytest.mark.parametrize("shift", [(0, 1), (3, 2), (7, 5)])
@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 1, 16, 16)])
def test_shift_invariance_of_energy(
    shape: tuple[int, ...],
    shift: tuple[int, int],
) -> None:
    """Circular shift does not change the Fourier energy ‖fft2c(roll(x))‖² == ‖fft2c(x)‖².

    This follows directly from Parseval + isometry of circular shift:
    ‖roll(x)‖² = ‖x‖² and ‖fft2c(·)‖² = ‖·‖².
    """
    B, C, H, W = shape
    p, q = shift
    seed = hash(("shift_energy", shape, shift)) & 0xFFFF_FFFF

    gen = torch.Generator()
    gen.manual_seed(seed)
    x = random_complex(shape, dtype=torch.complex64, generator=gen)

    energy_K = torch.sum(torch.abs(fft2c(x)) ** 2).item()
    x_shifted = torch.roll(x, shifts=(p, q), dims=(-2, -1))
    energy_K_shifted = torch.sum(torch.abs(fft2c(x_shifted)) ** 2).item()

    scale = max(abs(energy_K), abs(energy_K_shifted), 1e-8)
    rel_diff = abs(energy_K - energy_K_shifted) / scale

    tol = tol_for(torch.complex64) * 10
    assert rel_diff < tol, (
        f"Shift-invariance of energy violated for shape={shape}, shift={shift}.\n"
        f"  ‖K‖²={energy_K:.6g}  ‖K_shifted‖²={energy_K_shifted:.6g}  "
        f"rel_diff={rel_diff:.2e}  tol={tol:.2e}"
    )
