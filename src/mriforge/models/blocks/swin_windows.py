"""Windowing primitives for shifted-window (Swin) attention.

One owner for the three operations a Swin block needs: partition a feature map
into non-overlapping windows, reverse that partition, and build the attention
mask that stops a *shifted* window from attending across the cyclic wrap.

The mask belongs here, behind a resolution-keyed cache, because
:class:`~mriforge.models.blocks.swin.WindowAttention` derives its window count
from ``mask.shape[0]``. A mask built for one resolution and applied at another
does not merely mis-weight the attention -- it reshapes the attention tensor by
the wrong factor: silently when the two counts happen to divide, and with a
``RuntimeError`` when they do not (#1345).

The padding rule is here for the same reason. It was previously spelled twice
inside ``SwinBlock`` -- once when building the mask and once in ``forward`` --
and the mask is only valid if both spellings agree (non-negotiable 17).
"""

from __future__ import annotations

import torch

__all__ = [
    "ShiftedWindowMaskCache",
    "build_shifted_window_mask",
    "padded_resolution",
    "window_partition",
    "window_reverse",
]


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def padded_resolution(height: int, width: int, window_size: int) -> tuple[int, int]:
    """Round a resolution up to a whole number of windows.

    The single spelling of the rule. The mask is built on the padded grid and
    the feature map is padded to the same grid in ``SwinBlock.forward``; if the
    two disagree the mask has the wrong number of windows.
    """
    padded_h = height + (window_size - height % window_size) % window_size
    padded_w = width + (window_size - width % window_size) % window_size
    return padded_h, padded_w


def window_count(height: int, width: int, window_size: int) -> int:
    """Number of windows the padded ``(height, width)`` grid partitions into."""
    padded_h, padded_w = padded_resolution(height, width, window_size)
    return (padded_h // window_size) * (padded_w // window_size)


def build_shifted_window_mask(
    height: int,
    width: int,
    window_size: int,
    shift_size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build the SW-MSA attention mask for one resolution.

    ``height``/``width`` are the *unpadded* feature-map dimensions; the mask is
    built on the padded grid, matching what ``SwinBlock.forward`` partitions.

    Args:
        height: Unpadded feature-map height.
        width: Unpadded feature-map width.
        window_size: Side of the square attention window.
        shift_size: Cyclic shift applied before partitioning. Must be > 0 --
            an unshifted block needs no mask, and asking for one is a caller
            bug rather than a case to silently return ``None`` for.
        device: Device to build on.
        dtype: Floating dtype of the returned mask.

    Returns:
        ``(num_windows, window_size**2, window_size**2)``, ``0.0`` where two
        positions may attend to each other and ``-100.0`` where they may not.
    """
    if shift_size <= 0:
        raise ValueError(
            f"build_shifted_window_mask needs shift_size > 0, got {shift_size}; "
            "an unshifted window block must pass mask=None instead"
        )
    padded_h, padded_w = padded_resolution(height, width, window_size)
    img_mask = torch.zeros((1, padded_h, padded_w, 1), device=device, dtype=dtype)
    spans = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    region = 0
    for h in spans:
        for w in spans:
            img_mask[:, h, w, :] = region
            region += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)


class ShiftedWindowMaskCache:
    """Resolution-keyed store of SW-MSA masks.

    A Swin block may legitimately see more than one resolution -- a generator
    reused across patch sizes, a validation pass at full volume after training
    on patches. Rebuilding the mask every forward would put a Python loop and a
    handful of small allocations on the warm path (non-negotiable 9), so each
    distinct ``(resolution, device, dtype)`` is built once.

    Not an ``nn.Module`` and not a buffer: the mask is *derived* state, fully
    determined by the resolution and the block's window geometry. Keeping it
    out of ``state_dict`` is what stops a checkpoint from being locked to the
    resolution it was trained at.
    """

    def __init__(self, limit: int = 8) -> None:
        """Args: limit: distinct keys to retain before dropping the oldest."""
        self._limit = limit
        self._masks: dict[tuple, torch.Tensor] = {}

    def __len__(self) -> int:
        """Number of distinct masks currently held."""
        return len(self._masks)

    def clear(self) -> None:
        """Drop every cached mask."""
        self._masks.clear()

    def get(
        self,
        height: int,
        width: int,
        window_size: int,
        shift_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return the mask for this resolution, building it on first use."""
        key = (height, width, window_size, shift_size, str(device), dtype)
        mask = self._masks.get(key)
        if mask is not None:
            return mask
        mask = build_shifted_window_mask(
            height,
            width,
            window_size,
            shift_size,
            device=device,
            dtype=dtype,
        )
        if len(self._masks) >= self._limit:
            # Bounded rather than evicting cleverly: a block that genuinely
            # cycles through more than `limit` resolutions is pathological, and
            # an unbounded dict on a long run is a leak.
            self._masks.pop(next(iter(self._masks)))
        self._masks[key] = mask
        return mask
