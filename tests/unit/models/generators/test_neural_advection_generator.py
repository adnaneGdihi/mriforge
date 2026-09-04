"""Tests for NeuralAdvectionGenerator — the velocity-field exposure seam.

The hyperelastic-Jacobian incompressibility loss (m5/m9 arm) must regularise the
STN VELOCITY field, not the reconstructed magnitude image. For that the strategy
needs the field after a forward pass, so the generator must cache it. Regression
for the 2026-06 finding that the field was computed and discarded, so the loss
fell back to det(J) of the image (a meaningless constraint).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.neural_advection_generator import NeuralAdvectionGenerator


def test_forward_caches_deformation_field() -> None:
    gen = NeuralAdvectionGenerator(in_channels=2, out_channels=2)
    x = torch.randn(2, 2, 16, 16)
    marker = torch.randn(2)
    out = gen(x, marker_signal=marker)
    field = gen.last_deformation_field
    assert field is not None, "velocity field was discarded (the bug)"
    # [B, 2, H, W] displacement field (vx, vy) — NOT the image.
    assert field.shape == (2, 2, 16, 16)
    # The cached field is the one used to warp: differs from the output image.
    assert field.shape == out.shape  # both [B, 2, H, W] here, but distinct tensors
    assert field is not out
