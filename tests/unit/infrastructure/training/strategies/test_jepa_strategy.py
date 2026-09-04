"""Tests for the I-JEPA masking helper.

Review finding (2026-07-01): ``multi_block_mask`` sampled mask geometry on the
input's ``device`` (CUDA in real training) and ``.item()``-synced every scalar
draw — ~16 blocking host/device syncs per training step (performance.md). Mask
geometry is scalar bookkeeping; it now runs on CPU (where ``.item()`` is a free
host read) and the boolean masks are moved to ``device`` exactly once.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.infrastructure.training.strategies.jepa_strategy import (  # noqa: E402
    multi_block_mask,
)


def test_multi_block_mask_contract() -> None:
    ctx, targets = multi_block_mask(32, 24, n_targets=4)
    assert ctx.shape == (32, 24)
    assert ctx.dtype == torch.bool
    assert len(targets) == 4
    assert all(t.shape == (32, 24) and t.dtype == torch.bool for t in targets)
    # Context is the complement of the union of targets (up to the dilation
    # fallback), so at least some pixels are masked out of context.
    assert ctx.sum() < ctx.numel()


def test_multi_block_mask_returns_on_requested_device() -> None:
    """The masks must come back on the requested device (moved once at the end)."""
    dev = torch.device("cpu")
    ctx, targets = multi_block_mask(16, 16, n_targets=3, device=dev)
    assert ctx.device == dev
    assert all(t.device == dev for t in targets)


def test_multi_block_mask_target_count_scales() -> None:
    for n in (1, 2, 6):
        _, targets = multi_block_mask(20, 20, n_targets=n)
        assert len(targets) == n
