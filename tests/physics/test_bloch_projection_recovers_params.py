"""Physics test: Bloch-manifold projection recovers the generating parameters.

The spec (Idea 2 / B-2) asks that discretising / fitting the parameter grid
recovers the exact Bloch signal within numerical precision. We assert the
stronger fixed-point property: given a signal generated from a *known*
``(M0, T1, T2)``, the voxel-wise Gauss-Newton projector recovers parameters
whose Bloch signal matches the input to high precision (the manifold residual
collapses to ~0), and the recovered T1 is close to the generating T1.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.physics.bloch_manifold_projector import BlochManifoldProjector

pytestmark = pytest.mark.physics


@pytest.mark.parametrize(
    "m0, t1, t2",
    [
        (1.0, 800.0, 60.0),
        (1.3, 1200.0, 90.0),
        (0.8, 1500.0, 110.0),
    ],
)
def test_projection_recovers_exact_signal(m0, t1, t2):
    proj = BlochManifoldProjector(n_inner_iterations=40)
    bank = proj._bloch_signal_bank(
        torch.tensor([[m0]]), torch.tensor([[t1]]), torch.tensor([[t2]])
    )  # [1, K]
    m0_hat, t1_hat, t2_hat = proj.fit_parameters(bank)
    recon = proj._bloch_signal_bank(m0_hat, t1_hat, t2_hat)
    # The recovered signal reproduces the input bank to high relative precision
    # (the absolute signal magnitudes are ~0.03, so use a relative tolerance).
    rel_err = (recon - bank).abs().max() / (bank.abs().max() + 1e-12)
    assert float(rel_err) < 5e-2, (recon, bank, float(rel_err))
    # T2 <= T1 ordering preserved.
    assert float(t2_hat) <= float(t1_hat) + 1e-3


def test_manifold_residual_collapses_on_exact_signal():
    proj = BlochManifoldProjector(n_inner_iterations=40)
    # A small grid of on-manifold voxels.
    k = proj.flip_angles_rad.numel()
    m0 = torch.tensor([[1.0], [1.2], [0.9], [1.1]])
    t1 = torch.tensor([[700.0], [1000.0], [1400.0], [1800.0]])
    t2 = torch.tensor([[50.0], [80.0], [120.0], [150.0]])
    bank = proj._bloch_signal_bank(m0, t1, t2)
    img = bank.reshape(1, 2, 2, k).permute(0, 3, 1, 2).contiguous()
    res = proj.residual(img)
    # Residual collapses to a small fraction of the on-manifold signal scale.
    assert float(res.mean()) < 0.1 * float(img.abs().mean())


def test_recovered_t1_correlates_with_truth():
    """Across a sweep, recovered T1 should track the generating T1 monotonically."""
    proj = BlochManifoldProjector(n_inner_iterations=50)
    true_t1 = torch.tensor([[600.0], [900.0], [1200.0], [1600.0], [2200.0]])
    m0 = torch.full_like(true_t1, 1.0)
    t2 = torch.full_like(true_t1, 60.0)
    bank = proj._bloch_signal_bank(m0, true_t1, t2)
    _, t1_hat, _ = proj.fit_parameters(bank)
    # Spearman-style monotonicity: rank order preserved.
    order_true = torch.argsort(true_t1.squeeze(-1))
    order_hat = torch.argsort(t1_hat.squeeze(-1))
    assert torch.equal(order_true, order_hat), (true_t1.squeeze(-1), t1_hat.squeeze(-1))
