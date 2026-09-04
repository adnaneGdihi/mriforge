"""VF k-space operator models — trajectory-estimate exposure.

``diff_trajectory_opt`` exposes its estimated (Cartesian per-readout-line)
trajectory deviation so a *matched* trajectory-scoring seam can read it. The
estimate is NOT comparable to a spiral acquisition trajectory (see exp_vf_35).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.vf_kspace_operators import (
    DifferentiableTrajectoryOptimizer,
)


def test_diff_trajectory_opt_exposes_last_trajectory_estimate():
    m = DifferentiableTrajectoryOptimizer(in_channels=2, out_channels=2)
    x = torch.randn(1, 2, 32, 32)  # real-stacked complex
    out = m(x)
    assert out.shape == x.shape  # corrected image, same shape
    est = getattr(m, "last_trajectory_estimate", None)
    assert est is not None
    # [B, 2, H] per-readout-line (kx, ky) deviation — Cartesian, not spiral coords
    assert est.shape[0] == 1 and est.shape[1] == 2
    assert torch.isfinite(est).all()
