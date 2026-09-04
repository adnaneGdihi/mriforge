"""Sanity-shape tests for motion_correction.py over the project shape matrices.

CenterOfMassAlign and CrossCorrelationShift both take [B, T, H, W].
We map the SHAPE_MATRIX_2D (B, C, H, W) to (B, T=C, H, W) (treat C as T).
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id
from spectramr.infrastructure.physics.motion_correction import (
    CenterOfMassAlign,
    CrossCorrelationShift,
)


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, t, h, w",
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D],
)
def test_com_align_shape_matrix(b: int, t: int, h: int, w: int):
    aligner = CenterOfMassAlign()
    x = torch.rand(b, t, h, w)
    out = aligner(x)
    assert out.shape == (b, t, h, w)


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, t, h, w",
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D],
)
def test_xcorr_shift_shape_matrix(b: int, t: int, h: int, w: int):
    aligner = CrossCorrelationShift()
    x = torch.rand(b, t, h, w)
    out = aligner(x)
    assert out.shape == (b, t, h, w)
