r"""2D Discrete Cosine Transform (DCT-II / DCT-III) operations.

This module provides the type-II DCT (``dct2_neumann``) and its inverse,
the type-III DCT (``idct2_neumann``), implemented via the FFT
mirror-extension trick. The DCT-II is the basis of eigenfunctions of the
2D Laplacian :math:`\Delta` under Neumann (reflecting) boundary
conditions, which is exactly what the heat-equation forward process of
blurring diffusion requires (Hoogeboom & Salimans, ICLR 2023).

.. math::

    X_k = 2 \sum_{n=0}^{N-1} x_n \cos\!\left(\frac{\pi (2n+1) k}{2N}\right)

**Why DCT and not DFT here.** The DFT diagonalises :math:`\Delta` under
*periodic* boundary conditions, which introduces wraparound artefacts.
The DCT-II diagonalises :math:`\Delta` under *Neumann* conditions (no
mass flux across the image boundary), which matches the physics.

**AMP / k-space scope note.** These transforms operate on *real-valued*
feature maps with their own normalisation conventions. They are NOT
complex MRI k-space round-trips, so the "use ``fft2c`` / ``ifft2c``"
rule in ``CLAUDE.md`` does not apply (this is the explicit DCT/spectral
exemption documented in pitfall #2). To keep results numerically stable
under mixed precision, the FFTs are run inside an ``autocast``-disabled
region in ``float32`` and cast back to the caller's dtype on exit.
"""

from __future__ import annotations

import torch

__all__ = ["dct1d", "dct2_neumann", "idct1d", "idct2_neumann"]


def _autocast_off() -> torch.autocast:
    """Context manager that disables CUDA autocast for FFT-heavy code.

    Mirrors the pattern in ``fft_ops.py``: spectral transforms in half
    precision accumulate large round-off and can produce NaN gradients,
    so we force ``float32`` math inside the FFT region.
    """
    return torch.amp.autocast("cuda", enabled=False)


def dct1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1D type-II DCT along ``dim`` via the FFT mirror-extension trick.

    The orthonormal DCT-II is computed by even-mirror extending the
    signal, taking an FFT, and applying the standard twiddle factors.

    Args:
        x: Real-valued input tensor.
        dim: Axis along which to transform.

    Returns:
        Real-valued DCT-II coefficients with the same shape as ``x``.
    """
    in_dtype = x.dtype
    x = x.movedim(dim, -1)
    n = x.shape[-1]

    with _autocast_off():
        xf = x.float()
        # Even-mirror reorder: [x0, x2, ..., x_{N-1}, ..., x3, x1]
        v = torch.cat([xf[..., ::2], xf[..., 1::2].flip(-1)], dim=-1)
        vc = torch.fft.fft(v, dim=-1)

        k = torch.arange(n, dtype=torch.float32, device=x.device)
        twiddle = torch.exp(-1j * torch.pi * k / (2 * n))
        out = (vc * twiddle).real

        # Orthonormal (DCT-II "ortho") scaling: every coefficient is
        # multiplied by sqrt(2/N), with the DC term (k=0) additionally
        # divided by sqrt(2) so the basis is orthonormal.
        out = out * ((2.0 / n) ** 0.5)
        out[..., 0] = out[..., 0] / (2.0**0.5)

    out = out.movedim(-1, dim)
    return out.to(in_dtype)


def idct1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Inverse of :func:`dct1d` (orthonormal type-III DCT) along ``dim``.

    Args:
        x: Real-valued DCT-II coefficients.
        dim: Axis along which to invert.

    Returns:
        Real-valued reconstructed signal with the same shape as ``x``.
    """
    in_dtype = x.dtype
    x = x.movedim(dim, -1)
    n = x.shape[-1]

    with _autocast_off():
        xf = x.float().clone()
        # Undo the orthonormal scaling applied in dct1d to recover the real
        # part R[k] = Re(vc[k] * exp(-i*pi*k/2n)) of the twiddled spectrum.
        scale = (2.0 / n) ** 0.5
        xf = xf / scale
        xf[..., 0] = xf[..., 0] * (2.0**0.5)
        r = xf

        # The DCT-II keeps only the REAL part of vc[k]*exp(-i*pi*k/2n), so the
        # naive "vc = R*twiddle" reconstruction (which assumes the imaginary
        # part is zero) is NOT the inverse. Because the mirror-reordered signal
        # v is real, vc is conjugate-symmetric, which pins the discarded
        # imaginary part exactly: I[k] = -R[(n-k) mod n], with I[0] = 0. This
        # is the Makhoul half-spectrum reconstruction; without it the round
        # trip idct1d(dct1d(x)) is a (non-constant) ~0.45x distortion of x.
        r_mirror = torch.roll(torch.flip(r, dims=[-1]), shifts=1, dims=[-1])
        imag = -r_mirror
        imag[..., 0] = 0.0

        k = torch.arange(n, dtype=torch.float32, device=x.device)
        twiddle = torch.exp(1j * torch.pi * k / (2 * n))
        vc = torch.complex(r, imag) * twiddle
        v = torch.fft.ifft(vc, dim=-1).real

        out = torch.empty_like(v)
        half = (n + 1) // 2
        out[..., ::2] = v[..., :half]
        out[..., 1::2] = v[..., half:].flip(-1)

    out = out.movedim(-1, dim)
    return out.to(in_dtype)


def dct2_neumann(x: torch.Tensor) -> torch.Tensor:
    """2D orthonormal type-II DCT over the last two spatial axes.

    Applies :func:`dct1d` separably along ``H`` then ``W``. Preserves
    device, dtype, and leading ``[..., H, W]`` batch/channel dims.
    AMP-safe: the underlying FFTs run in ``float32`` with autocast off.

    Args:
        x: Real-valued tensor of shape ``[..., H, W]``.

    Returns:
        DCT-II coefficients of the same shape as ``x``.
    """
    x = dct1d(x, dim=-2)
    x = dct1d(x, dim=-1)
    return x


def idct2_neumann(x: torch.Tensor) -> torch.Tensor:
    """Inverse 2D DCT (type-III) over the last two spatial axes.

    Inverts :func:`dct2_neumann`; ``idct2_neumann(dct2_neumann(x)) == x``
    up to floating-point tolerance.

    Args:
        x: DCT-II coefficients of shape ``[..., H, W]``.

    Returns:
        Reconstructed real-valued tensor of the same shape as ``x``.
    """
    x = idct1d(x, dim=-1)
    x = idct1d(x, dim=-2)
    return x
