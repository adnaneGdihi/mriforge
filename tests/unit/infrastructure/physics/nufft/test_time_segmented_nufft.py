"""Unit tests for time-segmented multi-frequency-interpolation NUFFT.

Targets :mod:`spectramr.infrastructure.physics.nufft.time_segmented_nufft`.

Regression focus:

- ``_compute_interpolation_weights`` must NOT raise ``IndexError`` when there
  is a single time bin (``L == 1`` / ``num_time_segments == 1`` — a legitimate
  "no B0 segmentation" request). Pre-fix it indexed ``tau_bins[1]`` which is out
  of range for a length-1 ``linspace``. The fixed contract: a single bin yields
  constant unit weights of shape ``[1, N]``.
- ``L >= 2`` interpolation weights are normalised (column-sums ~ 1) — guards
  that the working path is unchanged.
- The constructor validates ``num_time_segments >= 1`` (raise, no silent
  fallback).

CPU-only and deterministic. The class constructs ``torchkbnufft`` operators, so
the whole module is skipped when ``torchkbnufft`` is unavailable.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torchkbnufft")

from spectramr.infrastructure.physics.nufft.time_segmented_nufft import (  # noqa: E402
    TimeSegmentedNUFFT,
)


def _make_op(num_time_segments: int = 6) -> TimeSegmentedNUFFT:
    return TimeSegmentedNUFFT(image_shape=(16, 16), num_time_segments=num_time_segments)


def test_single_time_bin_weights_no_indexerror() -> None:
    """L == 1: a single time bin must give constant unit weights [1, N],
    not an IndexError from ``tau_bins[1]``."""
    op = _make_op(num_time_segments=1)
    readout_times = torch.linspace(1e-3, 30e-3, 8)
    tau_bins = torch.linspace(float(readout_times.min()), float(readout_times.max()), 1)
    weights = op._compute_interpolation_weights(readout_times, tau_bins)
    assert weights.shape == (1, 8)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_multi_bin_weights_are_normalised() -> None:
    """L >= 2 (working path): column-sums of the interpolation weights ~ 1."""
    op = _make_op(num_time_segments=4)
    readout_times = torch.linspace(1e-3, 30e-3, 12)
    tau_bins = torch.linspace(float(readout_times.min()), float(readout_times.max()), 4)
    weights = op._compute_interpolation_weights(readout_times, tau_bins)  # [L, N]
    assert weights.shape == (4, 12)
    col_sums = weights.sum(dim=0)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4)


def test_num_time_segments_below_one_raises() -> None:
    """The constructor must raise on num_time_segments < 1 (no silent fallback)."""
    with pytest.raises(ValueError, match="num_time_segments must be >= 1"):
        TimeSegmentedNUFFT(image_shape=(16, 16), num_time_segments=0)


def test_l1_forward_runs_without_indexerror() -> None:
    """End-to-end: an L==1 operator forward must not crash on the weight
    computation path (the original IndexError surfaced through forward())."""
    op = _make_op(num_time_segments=1)
    B, H, W, K = 1, 16, 16, 24
    image = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
    b0_map = torch.zeros(B, 1, H, W)
    # 2-D trajectory in [-pi, pi].
    trajectory = (torch.rand(B, 2, K) - 0.5) * 2 * torch.pi
    out = op.forward(image=image, b0_map=b0_map, trajectory=trajectory)
    assert out.shape == (B, 1, K)
    assert torch.is_complex(out)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Regression: the advertised-but-unimplemented GNL knob fails loud (2026-07)
# ---------------------------------------------------------------------------
# gnl_displacement was accepted and silently ignored (the forward branch was a
# bare `pass`; the adjoint never referenced it) — an inert facade (pitfalls
# #16/#15). It now raises rather than pretending to fuse GNL.


def test_forward_gnl_displacement_raises() -> None:
    """Passing gnl_displacement to forward() must raise NotImplementedError."""
    op = _make_op(num_time_segments=2)
    B, H, W, K = 1, 16, 16, 24
    image = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
    b0_map = torch.zeros(B, 1, H, W)
    trajectory = (torch.rand(B, 2, K) - 0.5) * 2 * torch.pi
    gnl = torch.zeros(B, 2, H, W)
    with pytest.raises(NotImplementedError, match="gnl_displacement"):
        op.forward(image=image, b0_map=b0_map, trajectory=trajectory, gnl_displacement=gnl)


def test_adjoint_gnl_displacement_raises() -> None:
    """Passing gnl_displacement to adjoint() must raise NotImplementedError."""
    op = _make_op(num_time_segments=2)
    B, H, W, K = 1, 16, 16, 24
    kspace = torch.complex(torch.randn(B, 1, K), torch.randn(B, 1, K))
    b0_map = torch.zeros(B, 1, H, W)
    trajectory = (torch.rand(B, 2, K) - 0.5) * 2 * torch.pi
    gnl = torch.zeros(B, 2, H, W)
    with pytest.raises(NotImplementedError, match="gnl_displacement"):
        op.adjoint(kspace=kspace, b0_map=b0_map, trajectory=trajectory, gnl_displacement=gnl)


def test_forward_without_gnl_still_works() -> None:
    """The B0-only path (gnl_displacement=None) is unaffected."""
    op = _make_op(num_time_segments=3)
    B, H, W, K = 1, 16, 16, 24
    image = torch.complex(torch.randn(B, 1, H, W), torch.randn(B, 1, H, W))
    b0_map = torch.zeros(B, 1, H, W)
    trajectory = (torch.rand(B, 2, K) - 0.5) * 2 * torch.pi
    out = op.forward(image=image, b0_map=b0_map, trajectory=trajectory)
    assert out.shape == (B, 1, K) and torch.is_complex(out)
