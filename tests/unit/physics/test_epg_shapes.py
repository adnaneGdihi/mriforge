"""Sanity-shape tests for epg.py parametrised over the project shape matrices.

Uses SHAPE_MATRIX_2D as [B, C, H, W] where C=1 for all EPG inputs.
"""

from __future__ import annotations

import pytest
import torch

from tests.utils.shape_matrices import SHAPE_MATRIX_2D, shape_id
from spectramr.infrastructure.physics.epg import simulate_differentiable_epg_fse


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "b, c, h, w",
    # Force C=1 for EPG (C dim is always 1 in this module)
    [(b, c, h, w) for (b, c, h, w) in SHAPE_MATRIX_2D if c == 1],
    ids=[shape_id((b, c, h, w)) for (b, c, h, w) in SHAPE_MATRIX_2D if c == 1],
)
def test_diff_epg_shape_matrix(b: int, c: int, h: int, w: int):
    """differentiable EPG output matches input [B, 1, H, W] for all shape-matrix entries."""
    rho = torch.rand(b, c, h, w).clamp(min=0.05)
    t1  = torch.rand(b, c, h, w) * 900.0 + 100.0
    t2  = torch.rand(b, c, h, w) * 80.0  + 20.0
    fa_ex  = torch.full((b, c, h, w), 90.0)
    fa_ref = torch.full((b, c, h, w), 180.0)
    out = simulate_differentiable_epg_fse(rho, t1, t2, fa_ex, fa_ref, echo_train_length=8)
    assert out.shape == (b, c, h, w), f"shape mismatch: {tuple(out.shape)}"
    assert not torch.isnan(out).any(), "NaN in output"
