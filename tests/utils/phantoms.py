"""Deterministic CPU phantom generators for MRI test fixtures.

All generators accept an optional ``torch.Generator`` so callers can
produce reproducible tensors without globally seeding the RNG.

Shapes follow the project convention ``[B, C, H, W]`` unless a scalar
spatial size is requested (Shepp-Logan and point-source variants return
``[n, n]`` 2D arrays by default and are wrapped to ``[1,1,n,n]`` by
the caller if a batch dimension is needed).

Note: these phantoms are intentionally minimal and CPU-only.  They are
not meant to reproduce clinical MRI quality — they exist purely to give
test code deterministic, non-zero inputs with controlled spectral
content.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ellipse(
    n: int,
    center_x: float,
    center_y: float,
    semi_x: float,
    semi_y: float,
    angle_deg: float,
    intensity: float,
    out: torch.Tensor,
) -> None:
    """Rasterise one ellipse onto an ``[n, n]`` float32 tensor (in-place)."""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Coordinate grid in [-1, 1]
    ys = torch.linspace(-1.0, 1.0, n, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, n, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # Shift to ellipse centre
    dx = xx - center_x
    dy = yy - center_y

    # Rotate
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    # Ellipse membership
    inside = (xr / semi_x) ** 2 + (yr / semi_y) ** 2 <= 1.0
    out[inside] += intensity


# Standard 10-ellipse Shepp-Logan parameter table.
# Each row: (center_x, center_y, semi_x, semi_y, angle_deg, intensity)
_SHEPP_LOGAN_PARAMS: list[tuple[float, float, float, float, float, float]] = [
    (0.00,  0.00, 0.69, 0.92,  0.0,  2.00),
    (0.00, -0.01, 0.6624, 0.8740,  0.0, -0.98),
    (0.22,  0.00, 0.11, 0.31, -18.0, -0.02),
    (-0.22,  0.00, 0.16, 0.41,  18.0, -0.02),
    (0.00,  0.35, 0.21, 0.25,   0.0,  0.01),
    (0.00,  0.10, 0.046, 0.046, 0.0,  0.01),
    (0.00, -0.10, 0.046, 0.046, 0.0,  0.01),
    (-0.08, -0.605, 0.046, 0.023, 0.0, 0.01),
    (0.00, -0.605, 0.023, 0.023, 0.0, 0.01),
    (0.06, -0.605, 0.023, 0.046, 0.0, 0.01),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def shepp_logan_2d(n: int) -> torch.Tensor:
    """Return a deterministic analytic Shepp-Logan phantom.

    The phantom is computed analytically by summing filled ellipses —
    no external library (scipy, skimage) required.

    Args:
        n: Spatial resolution; the output is ``[n, n]`` float32.

    Returns:
        torch.Tensor: Float32 tensor of shape ``[n, n]``, values in
        roughly ``[-1, 2]`` clipped to ``[0, 1]`` after normalisation.
    """
    out = torch.zeros(n, n, dtype=torch.float32)
    for cx, cy, sx, sy, angle, val in _SHEPP_LOGAN_PARAMS:
        _ellipse(n, cx, cy, sx, sy, angle, val, out)
    # Normalise to [0, 1]
    vmin = out.min()
    vmax = out.max()
    if vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
    return out


def gaussian_blob(
    shape: Sequence[int],
    sigma: float | Sequence[float],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a deterministic Gaussian blob centred in *shape*.

    Args:
        shape: Spatial dimensions ``(H, W)`` or ``(D, H, W)`` — any
            positive-integer iterable.  The result has the same number
            of dimensions.
        sigma: Standard deviation in pixels.  Scalar applies to all
            axes; a sequence must match ``len(shape)``.
        generator: Optional RNG for centre jitter (unused here — the
            blob is always centred; the parameter is present for API
            symmetry).

    Returns:
        torch.Tensor: Float32 tensor of the requested shape, values in
        ``[0, 1]``.
    """
    shape = tuple(shape)
    ndim = len(shape)

    if isinstance(sigma, (int, float)):
        sigmas = [float(sigma)] * ndim
    else:
        sigmas = [float(s) for s in sigma]
        if len(sigmas) != ndim:
            raise ValueError(
                f"sigma has {len(sigmas)} elements but shape has {ndim} dims"
            )

    grids = [
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
        for size in shape
    ]
    meshes = torch.meshgrid(*grids, indexing="ij")

    # Normalise sigma to [-1, 1] range
    scaled_sigmas = [2.0 * s / max(size, 1) for s, size in zip(sigmas, shape)]

    exponent = sum(
        (g / (2.0 * sig)) ** 2 if sig > 0 else torch.zeros_like(g)
        for g, sig in zip(meshes, scaled_sigmas)
    )
    blob = torch.exp(-exponent)
    return blob


def point_source(
    shape: Sequence[int],
    k0: int | Sequence[int] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a delta function at position *k0* (or the centre if None).

    Args:
        shape: Spatial dimensions ``(H, W)`` or similar.
        k0: Index of the non-zero pixel.  May be a scalar (applied to
            all axes) or a sequence of length ``len(shape)``.  Defaults
            to the centre of each axis.
        generator: Unused; present for API symmetry.

    Returns:
        torch.Tensor: Float32 tensor of the requested shape with a
        single 1.0 at position *k0*, zeros elsewhere.
    """
    shape = tuple(shape)
    out = torch.zeros(shape, dtype=torch.float32)

    if k0 is None:
        idx: tuple[int, ...] = tuple(s // 2 for s in shape)
    elif isinstance(k0, int):
        idx = tuple(k0 % s for s in shape)
    else:
        idx = tuple(int(k) % s for k, s in zip(k0, shape))

    out[idx] = 1.0
    return out


def random_complex(
    shape: Sequence[int],
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a random complex tensor with unit-normal real and imaginary parts.

    Args:
        shape: Output shape (any number of dimensions).
        dtype: ``torch.complex64`` (fp32 components) or
            ``torch.complex128`` (fp64 components).  Defaults to
            ``complex64``.
        generator: Optional :class:`torch.Generator` for reproducibility.

    Returns:
        torch.Tensor: Complex tensor of the requested shape.
    """
    shape = tuple(shape)
    if dtype == torch.complex64:
        real_dtype = torch.float32
    elif dtype == torch.complex128:
        real_dtype = torch.float64
    else:
        raise ValueError(f"Unsupported complex dtype: {dtype}")

    real_part = torch.randn(shape, dtype=real_dtype, generator=generator)
    imag_part = torch.randn(shape, dtype=real_dtype, generator=generator)
    return torch.complex(real_part, imag_part)
