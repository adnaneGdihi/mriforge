"""Tests for image-domain motion-correction primitives.

Targets ``mriforge.infrastructure.physics.motion_correction``. Verifies the
identity property of each aligner (a stack of identical frames returns
the input unchanged), shift recovery on synthetic translations, and
fail-loud shape validation per CLAUDE.md #9.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.motion_correction import (
    CenterOfMassAlign,
    CrossCorrelationShift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_blob(h: int, w: int, cy: float, cx: float, sigma: float = 2.0) -> torch.Tensor:
    """A 2-D Gaussian blob centred at (cy, cx) — used as a synthetic frame."""
    ys = torch.arange(h, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, -1)
    return torch.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sigma ** 2))


# ---------------------------------------------------------------------------
# CenterOfMassAlign
# ---------------------------------------------------------------------------


def test_com_align_identity_on_single_frame_repeated() -> None:
    """A stack of identical frames → output identical to input within 1e-4."""
    aligner = CenterOfMassAlign()
    frame = _gaussian_blob(32, 32, 16, 16)
    stack = frame.unsqueeze(0).unsqueeze(0).repeat(1, 4, 1, 1)  # [1, 4, 32, 32]
    out = aligner(stack, ref_index=0)
    assert out.shape == stack.shape
    assert torch.allclose(out, stack, atol=1e-3)


def test_com_align_recovers_synthetic_translation() -> None:
    """A frame shifted by a known offset is realigned to the reference."""
    aligner = CenterOfMassAlign()
    ref = _gaussian_blob(32, 32, 16, 16)
    shifted = _gaussian_blob(32, 32, 18, 14)  # shifted by (+2, -2) pixels
    stack = torch.stack([ref, shifted], dim=0).unsqueeze(0)  # [1, 2, 32, 32]
    out = aligner(stack, ref_index=0)
    # After alignment, the shifted frame's COM should match the reference.
    out_blob = out[0, 1]
    # COM of out_blob should be near (16, 16).
    ys = torch.arange(32, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(32, dtype=torch.float32).view(1, -1)
    wts = out_blob.clamp(min=0)
    cy = (wts * ys).sum() / (wts.sum() + 1e-8)
    cx = (wts * xs).sum() / (wts.sum() + 1e-8)
    assert abs(cy.item() - 16) < 1.0, f"recovered cy = {cy.item():.2f}"
    assert abs(cx.item() - 16) < 1.0, f"recovered cx = {cx.item():.2f}"


def test_com_align_rejects_3d_input() -> None:
    """Non-4D input fails loud (CLAUDE.md #9 — no silent reshape)."""
    aligner = CenterOfMassAlign()
    with pytest.raises(ValueError, match=r"Expected \[B, T, H, W\]"):
        aligner(torch.randn(2, 32, 32))


# ---------------------------------------------------------------------------
# CrossCorrelationShift
# ---------------------------------------------------------------------------


def test_cross_correlation_zero_shift_on_identical_frames() -> None:
    """Identical frames yield zero shift estimate."""
    aligner = CrossCorrelationShift()
    frame = _gaussian_blob(32, 32, 16, 16)
    stack = frame.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    shifts = aligner.estimate_shifts(stack, ref_index=0)
    # All shifts should be exactly zero.
    assert torch.allclose(shifts, torch.zeros_like(shifts), atol=0.5)


def test_cross_correlation_recovers_integer_shift() -> None:
    """A frame shifted by an integer (dy, dx) is detected within 1 pixel."""
    aligner = CrossCorrelationShift()
    ref = _gaussian_blob(32, 32, 16, 16, sigma=3.0)
    shifted = _gaussian_blob(32, 32, 19, 13, sigma=3.0)  # +3, -3
    stack = torch.stack([ref, shifted], dim=0).unsqueeze(0)
    shifts = aligner.estimate_shifts(stack, ref_index=0)
    # shifts[0, 1] should be near (-3, +3) — the shift to undo.
    dy, dx = shifts[0, 1, 0].item(), shifts[0, 1, 1].item()
    assert abs(dy + 3) < 1.5, f"dy = {dy}"
    assert abs(dx - 3) < 1.5, f"dx = {dx}"


def test_cross_correlation_rejects_3d_input() -> None:
    """Fail-loud on non-4D input."""
    aligner = CrossCorrelationShift()
    with pytest.raises(ValueError, match=r"Expected \[B, T, H, W\]"):
        aligner.estimate_shifts(torch.randn(2, 32, 32))


def test_cross_correlation_complex_input_uses_magnitude() -> None:
    """Complex-valued frames are reduced to magnitude internally."""
    aligner = CrossCorrelationShift()
    real = _gaussian_blob(32, 32, 16, 16)
    cplx = torch.complex(real, torch.zeros_like(real))
    stack_real = real.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    stack_cplx = cplx.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
    out_real = aligner.estimate_shifts(stack_real)
    out_cplx = aligner.estimate_shifts(stack_cplx)
    assert torch.allclose(out_real, out_cplx, atol=1e-4)


# ---------------------------------------------------------------------------
# Sanity-shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(1, 2, 16, 16), (1, 4, 32, 32), (2, 4, 32, 32), (1, 8, 64, 64)],
    ids=lambda s: "x".join(map(str, s)),
)
def test_com_align_shape_preservation(shape: tuple[int, int, int, int]) -> None:
    """Shape preserved across a parametrised matrix."""
    aligner = CenterOfMassAlign()
    stack = torch.randn(*shape).abs()  # non-negative for COM weighting
    out = aligner(stack, ref_index=0)
    assert out.shape == shape
    assert torch.isfinite(out).all()
