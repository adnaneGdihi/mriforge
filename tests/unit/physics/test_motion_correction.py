"""Canary / parametrized / edge-case / expected-failure tests for motion_correction.py.

Covers:
  - CenterOfMassAlign
  - CrossCorrelationShift
  - RigidAlign2D
  - MotionCorrectedMean

All primitives accept [B, T, H, W] temporal stacks (except RigidAlign2D which
takes [B, C, H, W]) and are CPU-deterministic.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stack(b: int, t: int, h: int, w: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, t, h, w, generator=g)


# ---------------------------------------------------------------------------
# Canary tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_canary_motion_correction_imports():
    """All public symbols import cleanly."""
    from mriforge.infrastructure.physics.motion_correction import (
        CenterOfMassAlign,
        CrossCorrelationShift,
        RigidAlign2D,
        MotionCorrectedMean,
    )
    assert all([CenterOfMassAlign, CrossCorrelationShift, RigidAlign2D, MotionCorrectedMean])


@pytest.mark.physics
def test_canary_center_of_mass_align():
    """CenterOfMassAlign forward pass on trivial input."""
    from mriforge.infrastructure.physics.motion_correction import CenterOfMassAlign
    aligner = CenterOfMassAlign()
    x = _stack(1, 4, 16, 16)
    out = aligner(x)
    assert out.shape == (1, 4, 16, 16)


@pytest.mark.physics
def test_canary_xcorr_shift():
    """CrossCorrelationShift forward pass on trivial input."""
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    x = _stack(1, 4, 16, 16)
    out = aligner(x)
    assert out.shape == (1, 4, 16, 16)


@pytest.mark.physics
def test_canary_motion_corrected_mean_com():
    """MotionCorrectedMean(com) reduces temporal stack to single image."""
    from mriforge.infrastructure.physics.motion_correction import MotionCorrectedMean
    m = MotionCorrectedMean(method="com")
    x = _stack(2, 4, 16, 16)
    out = m(x)
    assert out.shape == (2, 16, 16)


# ---------------------------------------------------------------------------
# Parametrized — CenterOfMassAlign
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, t, h, w",
    [
        (1, 2, 16, 16),
        (2, 4, 32, 32),
        (1, 3, 16, 32),   # rectangular
        (2, 2, 64, 64),
    ],
    ids=["b1-t2-sq16", "b2-t4-sq32", "b1-t3-rect", "b2-t2-sq64"],
)
def test_com_align_shape(b, t, h, w):
    from mriforge.infrastructure.physics.motion_correction import CenterOfMassAlign
    aligner = CenterOfMassAlign()
    x = _stack(b, t, h, w)
    out = aligner(x)
    assert out.shape == (b, t, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_com_align_identity_already_aligned():
    """Perfectly aligned stack: output should equal input (or close)."""
    from mriforge.infrastructure.physics.motion_correction import CenterOfMassAlign
    aligner = CenterOfMassAlign()
    # Single-frame stack: COM shift is always 0
    x = _stack(1, 1, 16, 16)
    out = aligner(x, ref_index=0)
    assert torch.allclose(out.float(), x.float(), atol=1e-4)


# ---------------------------------------------------------------------------
# Parametrized — CrossCorrelationShift
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, t, h, w",
    [
        (1, 2, 16, 16),
        (2, 3, 32, 32),
        (1, 2, 16, 32),   # rectangular
    ],
    ids=["b1-t2", "b2-t3", "b1-t2-rect"],
)
def test_xcorr_shift_shape(b, t, h, w):
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    x = _stack(b, t, h, w)
    out = aligner(x)
    assert out.shape == (b, t, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_xcorr_estimate_shifts_shape():
    """estimate_shifts returns [B, T, 2]."""
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    b, t, h, w = 2, 4, 16, 16
    x = _stack(b, t, h, w)
    shifts = aligner.estimate_shifts(x, ref_index=0)
    assert shifts.shape == (b, t, 2)


# ---------------------------------------------------------------------------
# Parametrized — RigidAlign2D
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize(
    "b, c, h, w",
    [
        (1, 1, 16, 16),
        (2, 1, 32, 32),
        (1, 2, 16, 32),
    ],
    ids=["b1-c1", "b2-c1", "b1-c2-rect"],
)
def test_rigid_align2d_identity(b, c, h, w):
    """RigidAlign2D with identity theta is a no-op."""
    from mriforge.infrastructure.physics.motion_correction import RigidAlign2D
    aligner = RigidAlign2D(learnable=False)
    image = torch.rand(b, c, h, w)
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).expand(b, -1, -1)
    out = aligner(image, theta=theta)
    assert out.shape == (b, c, h, w)
    assert torch.allclose(out, image, atol=1e-4)


# ---------------------------------------------------------------------------
# Parametrized — MotionCorrectedMean
# ---------------------------------------------------------------------------

@pytest.mark.physics
@pytest.mark.parametrize("method", ["com", "xcorr"])
def test_motion_corrected_mean_shapes(method):
    from mriforge.infrastructure.physics.motion_correction import MotionCorrectedMean
    m = MotionCorrectedMean(method=method)
    b, t, h, w = 2, 4, 16, 16
    x = _stack(b, t, h, w)
    out = m(x)
    assert out.shape == (b, h, w)
    assert not torch.isnan(out).any()


@pytest.mark.physics
def test_motion_corrected_mean_rigid():
    from mriforge.infrastructure.physics.motion_correction import MotionCorrectedMean
    m = MotionCorrectedMean(method="rigid")
    b, t, h, w = 2, 3, 16, 16
    x = _stack(b, t, h, w)
    # Supply per-frame identity theta: [B*T, 2, 3]
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).expand(b * t, -1, -1)
    out = m(x, theta=theta)
    assert out.shape == (b, h, w)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_com_align_edge_all_zeros():
    """All-zero input: COM is at centre; output should be all zeros."""
    from mriforge.infrastructure.physics.motion_correction import CenterOfMassAlign
    aligner = CenterOfMassAlign()
    x = torch.zeros(1, 2, 16, 16)
    out = aligner(x)
    assert not torch.isnan(out).any()
    assert torch.allclose(out, x, atol=1e-6)


@pytest.mark.physics
def test_xcorr_edge_single_frame():
    """T=1: shift vs itself should be ~zero."""
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    x = _stack(1, 1, 16, 16)
    shifts = aligner.estimate_shifts(x, ref_index=0)
    # shift of the reference frame vs itself = 0
    assert torch.allclose(shifts[:, 0, :], torch.zeros(1, 2), atol=1e-5)


@pytest.mark.physics
def test_xcorr_edge_complex_input():
    """CrossCorrelationShift handles complex (takes abs internally)."""
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    x = torch.randn(1, 2, 16, 16) + 1j * torch.randn(1, 2, 16, 16)
    out = aligner(x)
    assert out.shape == (1, 2, 16, 16)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Expected-failure (raises) tests
# ---------------------------------------------------------------------------

@pytest.mark.physics
def test_com_align_raises_wrong_rank():
    """CenterOfMassAlign requires 4-D input; 3-D should raise ValueError."""
    from mriforge.infrastructure.physics.motion_correction import CenterOfMassAlign
    aligner = CenterOfMassAlign()
    x = torch.rand(4, 16, 16)   # missing B dim
    with pytest.raises(ValueError):
        aligner(x)


@pytest.mark.physics
def test_xcorr_raises_wrong_rank():
    """CrossCorrelationShift requires 4-D input."""
    from mriforge.infrastructure.physics.motion_correction import CrossCorrelationShift
    aligner = CrossCorrelationShift()
    with pytest.raises(ValueError):
        aligner(torch.rand(4, 16, 16))


@pytest.mark.physics
def test_rigid_align2d_raises_wrong_theta_shape():
    """RigidAlign2D raises ValueError for theta with wrong last dims."""
    from mriforge.infrastructure.physics.motion_correction import RigidAlign2D
    aligner = RigidAlign2D(learnable=False)
    image = torch.rand(2, 1, 16, 16)
    bad_theta = torch.eye(3).unsqueeze(0).expand(2, -1, -1)  # [2, 3, 3] not [2, 2, 3]
    with pytest.raises(ValueError):
        aligner(image, theta=bad_theta)


@pytest.mark.physics
def test_rigid_align2d_raises_wrong_image_rank():
    """RigidAlign2D requires [B, C, H, W]."""
    from mriforge.infrastructure.physics.motion_correction import RigidAlign2D
    aligner = RigidAlign2D(learnable=False)
    image = torch.rand(1, 16, 16)    # 3-D — missing C
    theta = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0)
    with pytest.raises(ValueError):
        aligner(image, theta=theta)


@pytest.mark.physics
def test_motion_corrected_mean_rigid_raises_without_theta():
    """MotionCorrectedMean(rigid) must raise ValueError when theta=None."""
    from mriforge.infrastructure.physics.motion_correction import MotionCorrectedMean
    m = MotionCorrectedMean(method="rigid")
    x = _stack(1, 2, 8, 8)
    with pytest.raises(ValueError, match="theta"):
        m(x)


@pytest.mark.physics
def test_motion_corrected_mean_raises_invalid_method():
    """MotionCorrectedMean must raise ValueError for unknown method string."""
    from mriforge.infrastructure.physics.motion_correction import MotionCorrectedMean
    with pytest.raises(ValueError, match="method"):
        MotionCorrectedMean(method="invalid_method")
