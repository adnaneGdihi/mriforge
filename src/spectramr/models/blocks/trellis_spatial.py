"""TRELLIS spatial utilities for patch operations.

Migrated from src/implementations/trellis/modules/spatial.py during consolidation.
"""

from __future__ import annotations

import torch


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert an image/volume into non-overlapping patches."""

    dim = x.dim() - 2
    for axis in range(2, dim + 2):
        if x.shape[axis] % patch_size != 0:
            raise ValueError(
                f"Dimension {axis} must be divisible by patch_size; "
                f"got {x.shape[axis]} and {patch_size}",
            )

    # Validate spatial dimensions
    if dim == 2:
        _ = x.shape[2], x.shape[3]  # H, W validation
    elif dim == 3:
        _ = x.shape[2], x.shape[3], x.shape[4]  # H, W, D validation
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

    shape = []
    for axis in range(2, dim + 2):
        shape.extend([x.shape[axis] // patch_size, patch_size])

    x = x.reshape(*x.shape[:2], *shape)
    permute_order = [0]  # batch
    # Add spatial dimensions (every other starting from index 2)
    permute_order.extend(range(2, 2 + 2 * dim, 2))
    # Add channel dimension
    permute_order.append(1)
    # Add patch dimensions (every other starting from index 3)
    permute_order.extend(range(3, 3 + 2 * dim, 2))
    x = x.permute(*permute_order)
    return x.reshape(x.shape[0], *x.shape[1 : 1 + dim], x.shape[1 + dim] * (patch_size**dim))


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reconstruct a tensor from non-overlapping patches."""

    dim = x.dim() - 2
    divisor = patch_size**dim
    if x.shape[-1] % divisor != 0:
        raise ValueError(
            f"Last dimension must be divisible by patch_size ** dim; got {x.shape[-1]} and {divisor}",
        )

    if dim == 2:
        # Reshape to (batch, h//p, w//p, channels, p, p)
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3] // divisor,
            patch_size,
            patch_size,
        )
        # Permute to (batch, channels, h//p, p, w//p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)
        # Reshape to (batch, channels, h, w)
        h = x.shape[2] * patch_size
        w = x.shape[4] * patch_size
        return x.reshape(x.shape[0], x.shape[1], h, w)
    elif dim == 3:
        # Reshape to (batch, h//p, w//p, d//p, channels, p, p, p)
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            x.shape[3],
            x.shape[4] // divisor,
            patch_size,
            patch_size,
            patch_size,
        )
        # Permute to (batch, channels, h//p, p, w//p, p, d//p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # Reshape to (batch, channels, h, w, d)
        h = x.shape[2] * patch_size
        w = x.shape[4] * patch_size
        d = x.shape[6] * patch_size
        return x.reshape(x.shape[0], x.shape[1], h, w, d)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


__all__ = ["patchify", "unpatchify"]
