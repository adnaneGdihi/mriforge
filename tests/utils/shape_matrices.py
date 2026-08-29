"""Shared shape and dtype matrices for sanity-shape tests.

Sanity-shape tests parametrise over the documented input contract for
each registered layer / model / loss. Reusing these matrices keeps the
contract uniform across the test suite — adding a new spatial size or
batch size only takes editing one file.

Convention:
- ``SHAPE_MATRIX_2D`` covers the standard ``[B, C, H, W]`` MRI image
  layout used by image-domain models.
- ``SHAPE_MATRIX_KSPACE`` covers multi-coil k-space ``[B, Nc, H, W]``
  with realistic coil counts.
- Use ``pytest.mark.parametrize`` with ``ids=`` to keep test names
  readable.
"""

from __future__ import annotations

import torch

SHAPE_MATRIX_2D: list[tuple[int, int, int, int]] = [
    (1, 1, 32, 32),
    (1, 2, 32, 32),
    (2, 1, 64, 64),
    (2, 2, 64, 32),
    (4, 1, 128, 128),
    (1, 1, 256, 256),
]

SHAPE_MATRIX_3D: list[tuple[int, int, int, int, int]] = [
    (1, 1, 16, 32, 32),
    (1, 2, 8, 64, 64),
    (2, 1, 4, 128, 128),
]

SHAPE_MATRIX_KSPACE: list[tuple[int, int, int, int]] = [
    (1, 1, 64, 64),
    (1, 4, 64, 64),
    (1, 8, 64, 64),
    (2, 4, 128, 128),
    (1, 16, 32, 32),
]

SHAPE_MATRIX_TEMPORAL: list[tuple[int, int, int, int]] = [
    (1, 4, 32, 32),
    (2, 8, 32, 32),
    (1, 16, 64, 64),
]

DTYPE_MATRIX_REAL: tuple[torch.dtype, ...] = (
    torch.float32,
    torch.float16,
)

DTYPE_MATRIX_COMPLEX: tuple[torch.dtype, ...] = (
    torch.complex64,
    torch.complex128,
)


def shape_id(shape: tuple[int, ...]) -> str:
    """Generate a readable test ID from a shape tuple."""
    return "x".join(str(d) for d in shape)


def dtype_id(dtype: torch.dtype) -> str:
    """Generate a readable test ID from a dtype."""
    return str(dtype).replace("torch.", "")
