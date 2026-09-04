"""Differential test: ``fft2c`` / ``ifft2c`` vs SigPy centered FFT.

Verifies that the framework's canonical FFT operations produce numerically
identical results to SigPy's ``sp.fft(x, axes=(-2,-1), center=True,
norm="ortho")`` across a range of shapes — including odd and prime spatial
dimensions where fftshift behaviour is most sensitive.

Reference convention
--------------------
SigPy performs centered FFT as:
    y = sp.fft(x, axes=(-2,-1), center=True, norm="ortho")
which is equivalent to::
    ifftshift → fft2 → fftshift   (forward)
    ifftshift → ifft2 → fftshift  (inverse)

Our ``fft2c`` / ``ifft2c`` follow the same convention and upcast inputs to
``complex64``.  The tolerance is therefore matched to ``complex64`` precision
(atol ≈ 1e-5).

Markers
-------
- ``differential``   : cross-implementation comparison
- ``requires_sigpy`` : SigPy must be importable
- ``physics``        : physics-correctness test
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Guard: skip the whole module at collection time if sigpy is absent.
# ``pytest.importorskip`` raises ``pytest.skip.Exception`` during collection
# which pytest converts to a "skipped" status — no collection error.
# ---------------------------------------------------------------------------
sp = pytest.importorskip("sigpy")

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c  # noqa: E402

# ---------------------------------------------------------------------------
# Test shapes — regular, odd, prime dimensions to exercise fftshift paths
# ---------------------------------------------------------------------------
_SHAPES: list[tuple[int, ...]] = [
    (1, 1, 32, 32),       # baseline square
    (2, 1, 64, 128),      # rectangular
    (1, 1, 33, 33),       # odd square
    (1, 1, 97, 53),       # prime × prime  ← most sensitive
    (2, 4, 48, 48),       # multi-channel batch
    (1, 1, 16, 16),       # small, power-of-two
]

# Tolerance: complex64 ~ 6 significant decimal digits, but float32 FFT
# accumulates error across N; 1e-5 is the agreed ceiling.
_ATOL = 1e-5
_RTOL = 1e-5


def _complex_rand(shape: tuple[int, ...], seed: int = 42) -> torch.Tensor:
    """Return a seeded complex64 random tensor of the given shape."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    real = torch.randn(*shape, generator=rng)
    imag = torch.randn(*shape, generator=rng)
    return torch.complex(real, imag).to(torch.complex64)


def _sigpy_fft2c(x_np: np.ndarray) -> np.ndarray:
    """Forward centered 2-D FFT via SigPy (axes=-2,-1, ortho norm, centered).

    SigPy ``sp.fft`` may operate on complex64 or complex128 numpy arrays.
    We cast to complex64 to match the framework precision.
    """
    # SigPy works on numpy arrays.
    return sp.fft(x_np.astype(np.complex64), axes=(-2, -1), center=True, norm="ortho")


def _sigpy_ifft2c(k_np: np.ndarray) -> np.ndarray:
    """Inverse centered 2-D FFT via SigPy."""
    return sp.ifft(k_np.astype(np.complex64), axes=(-2, -1), center=True, norm="ortho")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.differential
@pytest.mark.requires_sigpy
@pytest.mark.physics
@pytest.mark.parametrize("shape", _SHAPES, ids=[str(s) for s in _SHAPES])
def test_fft2c_matches_sigpy_forward(shape: tuple[int, ...]) -> None:
    """fft2c(x) must match sp.fft(x, center=True, norm='ortho') to atol=1e-5."""
    x = _complex_rand(shape, seed=7)

    # Framework result (torch)
    k_fw = fft2c(x)  # complex64 tensor

    # SigPy reference (numpy)
    k_ref_np = _sigpy_fft2c(x.numpy())

    k_fw_np = k_fw.detach().numpy()
    assert k_fw_np.dtype == np.complex64, (
        f"fft2c output must be complex64, got {k_fw_np.dtype}"
    )

    max_abs_err = np.abs(k_fw_np - k_ref_np).max()
    assert max_abs_err < _ATOL, (
        f"fft2c vs sp.fft mismatch for shape {shape}: "
        f"max_abs_err={max_abs_err:.3e} (threshold {_ATOL:.1e})"
    )


@pytest.mark.differential
@pytest.mark.requires_sigpy
@pytest.mark.physics
@pytest.mark.parametrize("shape", _SHAPES, ids=[str(s) for s in _SHAPES])
def test_ifft2c_matches_sigpy_inverse(shape: tuple[int, ...]) -> None:
    """ifft2c(k) must match sp.ifft(k, center=True, norm='ortho') to atol=1e-5."""
    k = _complex_rand(shape, seed=13)

    # Framework result
    x_fw = ifft2c(k)

    # SigPy reference
    x_ref_np = _sigpy_ifft2c(k.numpy())

    x_fw_np = x_fw.detach().numpy()
    max_abs_err = np.abs(x_fw_np - x_ref_np).max()
    assert max_abs_err < _ATOL, (
        f"ifft2c vs sp.ifft mismatch for shape {shape}: "
        f"max_abs_err={max_abs_err:.3e} (threshold {_ATOL:.1e})"
    )


@pytest.mark.differential
@pytest.mark.requires_sigpy
@pytest.mark.physics
@pytest.mark.parametrize("shape", _SHAPES, ids=[str(s) for s in _SHAPES])
def test_fft2c_ifft2c_roundtrip_vs_sigpy(shape: tuple[int, ...]) -> None:
    """ifft2c(fft2c(x)) ≈ x AND matches the sigpy roundtrip to atol=1e-5."""
    x = _complex_rand(shape, seed=99)

    # Framework roundtrip
    x_rt = ifft2c(fft2c(x))

    # SigPy roundtrip
    x_np = x.numpy()
    x_ref_rt_np = _sigpy_ifft2c(_sigpy_fft2c(x_np))

    # Framework roundtrip self-consistency
    rt_err = (x_rt - x).abs().max().item()
    assert rt_err < _ATOL, (
        f"fft2c→ifft2c roundtrip error for shape {shape}: {rt_err:.3e}"
    )

    # Framework vs SigPy roundtrip
    cross_err = np.abs(x_rt.detach().numpy() - x_ref_rt_np).max()
    assert cross_err < _ATOL, (
        f"roundtrip cross-check vs sigpy for shape {shape}: {cross_err:.3e}"
    )


@pytest.mark.differential
@pytest.mark.requires_sigpy
@pytest.mark.physics
def test_fft2c_prime_spatial_dims() -> None:
    """Explicit prime-dim test: (1,1,97,53) — exercises non-power-of-two fftshift."""
    shape = (1, 1, 97, 53)
    x = _complex_rand(shape, seed=2718)

    k_fw_np = fft2c(x).detach().numpy()
    k_ref_np = _sigpy_fft2c(x.numpy())

    err = np.abs(k_fw_np - k_ref_np).max()
    assert err < _ATOL, f"prime-dim test failed: max_abs_err={err:.3e}"


@pytest.mark.differential
@pytest.mark.requires_sigpy
@pytest.mark.physics
def test_fft2c_output_dtype_is_complex64() -> None:
    """fft2c always produces complex64 regardless of input dtype."""
    shapes_and_dtypes = [
        ((1, 1, 32, 32), torch.complex64),
        # real input → _to_complex casts to complex64
        ((1, 1, 32, 32), torch.float32),
    ]
    for shape, dtype in shapes_and_dtypes:
        if dtype == torch.float32:
            x = torch.randn(*shape)
        else:
            x = _complex_rand(shape)
        k = fft2c(x)
        assert k.dtype == torch.complex64, (
            f"Expected complex64 output for input dtype={dtype}, got {k.dtype}"
        )
