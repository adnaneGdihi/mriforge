"""Hermitian-symmetry utilities for complex k-space tensors.

A real-valued image x has k-space K(k) = X(k) satisfying the Hermitian
symmetry K(-k) = conj(K(k)). When a network or noise process produces
k-space tensors that violate this symmetry, the inverse FFT picks up a
spurious imaginary component. This module provides:

- ``enforce_hermitian(K)`` — symmetrise a centered k-space tensor.
- ``is_hermitian(K, atol)`` — boolean check.
- ``hermitian_gaussian_noise(shape, ...)`` — sample noise that already
  satisfies Hermitian symmetry (for Idea 2, score-field tomography).

Convention:
    Tensors are centered k-space [..., H, W] (DC at H//2, W//2) and may
    be either ``torch.complex64`` / ``torch.complex128``, or
    real-valued with the last dimension being 2 (interleaved real/imag).
    The functions detect the layout automatically and return the same.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _to_complex(K: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Coerce input to complex; report whether we converted from (re, im) pair."""
    if torch.is_complex(K):
        return K, False
    if K.shape[-1] == 2:
        return torch.view_as_complex(K.contiguous()), True
    raise TypeError(
        f"hermitian utilities expect complex tensor or [..., 2] real tensor, "
        f"got dtype={K.dtype} shape={tuple(K.shape)}"
    )


def _from_complex(K: torch.Tensor, was_interleaved: bool) -> torch.Tensor:
    return torch.view_as_real(K) if was_interleaved else K


def _cyclic_mirror(Z: torch.Tensor) -> torch.Tensor:
    """Return ``Z[..., -i, -j]`` modulo the spatial dimensions.

    Centered spectra put DC at index ``N // 2``, so index ``i`` carries frequency
    ``i - N // 2`` and its conjugate partner sits at ``N // 2 - (i - N // 2)``:

    * **Even** ``N``: ``shift = 1`` — the partner of ``i`` is ``(N - i) mod N``.
      ``+Nyquist`` has no partner in range and wraps onto itself at index 0.
    * **Odd** ``N``: ``shift = 0`` — every frequency has a partner in range, so
      the map is the plain flip ``N - 1 - i``.

    The odd shift was ``2``, which compensated for ``fft2c`` mis-centering odd
    axes by one index (it applied ``fftshift`` where ``ifftshift`` was needed).
    Bending the mirror to match a broken SSOT made ``is_hermitian`` agree with
    ``fft2c`` while both disagreed with the centered-spectrum definition. The
    SSOT is fixed; the mirror follows the definition again.
    """
    h, w = Z.shape[-2], Z.shape[-1]
    shift_h = 1 if (h % 2 == 0) else 0
    shift_w = 1 if (w % 2 == 0) else 0
    return torch.roll(torch.flip(Z, dims=(-2, -1)), shifts=(shift_h, shift_w), dims=(-2, -1))


def enforce_hermitian(K: torch.Tensor) -> torch.Tensor:
    """Project centered k-space onto the Hermitian-symmetric subspace.

    Computes ``K_sym(k) = 0.5 * (K(k) + conj(K(-k)))`` using the cyclic
    mirror, so that the inverse FFT of ``K_sym`` is purely real (to
    numerical precision).

    Args:
        K: Centered k-space tensor [..., H, W], complex or [..., 2] real.

    Returns:
        Hermitian-projected tensor with the same dtype/shape as input.
    """
    Z, was_interleaved = _to_complex(K)
    Z_neg = _cyclic_mirror(Z)
    Z_sym = 0.5 * (Z + Z_neg.conj())
    return _from_complex(Z_sym, was_interleaved)


def is_hermitian(K: torch.Tensor, atol: float = 1e-5) -> bool:
    """Return True if K satisfies K(-k) ≈ conj(K(k)) within atol."""
    Z, _ = _to_complex(K)
    Z_neg = _cyclic_mirror(Z)
    residual = (Z - Z_neg.conj()).abs().max().item()
    return residual <= atol


def hermitian_gaussian_noise(
    shape: Sequence[int],
    *,
    dtype: torch.dtype = torch.complex64,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a centered k-space noise tensor that is Hermitian by construction.

    The forward-diffusion step in Idea 2 (score-field tomography) operates
    in centered k-space. Adding standard complex Gaussian noise breaks
    Hermitian symmetry; the inverse FFT then leaks energy into the
    imaginary channel. This sampler symmetrises a single i.i.d. complex
    Gaussian draw so the result lives on the same subspace as the data.

    Args:
        shape: Output shape [..., H, W].
        dtype: Complex dtype (``torch.complex64`` or ``torch.complex128``).
        device: Target device.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        Complex tensor of the requested shape with Hermitian symmetry.
    """
    if dtype not in (torch.complex64, torch.complex128):
        raise ValueError(f"hermitian_gaussian_noise requires complex dtype, got {dtype}")
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(*shape, dtype=real_dtype, device=device, generator=generator)
    imag = torch.randn(*shape, dtype=real_dtype, device=device, generator=generator)
    z = torch.complex(real, imag)
    return enforce_hermitian(z)


__all__ = ["enforce_hermitian", "hermitian_gaussian_noise", "is_hermitian"]
