"""Unit tests for the NUFFT forward/adjoint operator (B0 time-segmentation).

Targets :mod:`mriforge.infrastructure.physics.implementations.nufft_operator`.

Regression focus: ``_adjoint_b0_corrected`` computes ``seg_len = n_points //
n_segments``. When ``n_segments > n_points`` (short readout / tiny test),
``seg_len`` floors to 0 and every segment except the final one becomes an
empty-slice NUFFT call, silently collapsing the time-segmentation to a single,
mis-centered segment. The fixed contract: raise ``ValueError`` instead of
silently degrading. The valid path (``n_segments <= n_points``) still runs.

The operator constructs ``torchkbnufft`` operators, so the module is skipped
when ``torchkbnufft`` is unavailable. CPU-only, tiny tensors, fixed seed.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchkbnufft")

from mriforge.infrastructure.physics.implementations.nufft_operator import (  # noqa: E402
    NUFFTOperator,
)


def _make_op(n_segments: int, K: int):
    torch.manual_seed(0)
    traj = (torch.rand(2, K) - 0.5) * 2 * torch.pi
    return NUFFTOperator(traj=traj, im_size=(16, 16), n_segments=n_segments)


def test_b0_adjoint_raises_when_segments_exceed_points() -> None:
    """n_segments > n_points must raise, not silently collapse segmentation."""
    K = 4
    op = _make_op(n_segments=6, K=K)
    B, H, W = 1, 16, 16
    kspace = torch.complex(torch.randn(B, 1, K), torch.randn(B, 1, K))
    b0_map = torch.zeros(B, 1, H, W)
    with pytest.raises(ValueError, match="n_segments <= n_points"):
        op._adjoint_b0_corrected(kspace, smaps=None, b0_map=b0_map)


def test_b0_adjoint_valid_segments_runs() -> None:
    """n_segments <= n_points (the working path) reconstructs a finite image."""
    K = 24
    op = _make_op(n_segments=3, K=K)
    B, H, W = 1, 16, 16
    kspace = torch.complex(torch.randn(B, 1, K), torch.randn(B, 1, K))
    b0_map = torch.zeros(B, 1, H, W)
    recon = op._adjoint_b0_corrected(kspace, smaps=None, b0_map=b0_map)
    assert recon.shape == (B, 1, H, W)
    assert torch.is_complex(recon)
    assert torch.isfinite(recon).all()


def test_b0_adjoint_via_public_adjoint_raises() -> None:
    """The guard also fires through the public ``adjoint`` path when a b0_map
    is supplied and n_segments > n_points."""
    K = 4
    op = _make_op(n_segments=6, K=K)
    B, H, W = 1, 16, 16
    kspace = torch.complex(torch.randn(B, 1, K), torch.randn(B, 1, K))
    b0_map = torch.zeros(B, 1, H, W)
    with pytest.raises(ValueError, match="n_segments <= n_points"):
        op.adjoint(kspace, b0_map=b0_map)
