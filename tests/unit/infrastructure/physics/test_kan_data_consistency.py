"""Unit tests for KAN-based adaptive data consistency.

Verifies the KAN trust map is API-compatible with the existing
``AdaptiveDataConsistency`` (so it's a drop-in replacement under
``dc_method='kan_adaptive'``), respects the sampling mask, runs in both
image and k-space domain modes, and propagates gradients end-to-end.

CPU-only, small spatial sizes — runs in well under a second.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.physics.kan_data_consistency import KANAdaptiveDataConsistency


@pytest.fixture
def dc_layer():
    return KANAdaptiveDataConsistency(kan_hidden=8, kan_grid_size=4, kan_spline_order=3)


class TestKANAdaptiveDataConsistency:
    def test_no_op_when_measurements_missing(self, dc_layer):
        """Without ``measured_kspace`` or ``mask``, the layer is identity."""
        x = torch.randn(1, 4, 8, 8)
        out = dc_layer(x)
        assert torch.equal(out, x)
        out2 = dc_layer(x, measured_kspace=torch.randn(1, 4, 8, 8))  # mask missing
        assert torch.equal(out2, x)

    def test_kspace_domain_path_preserves_shape(self, dc_layer):
        """is_kspace_domain=True skips the FFT and returns same-shape k-space."""
        kspace = torch.randn(1, 4, 8, 8)
        measured = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 3:5, 3:5] = 1.0
        out = dc_layer(kspace, measured_kspace=measured, mask=mask, is_kspace_domain=True)
        assert out.shape == kspace.shape

    def test_image_domain_path_preserves_shape(self, dc_layer):
        """Image-domain input -> image-domain output."""
        img = torch.randn(1, 4, 8, 8)
        measured = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 3:5, 3:5] = 1.0
        out = dc_layer(img, measured_kspace=measured, mask=mask, is_kspace_domain=False)
        assert out.shape == img.shape

    def test_lambda_shape_and_range(self, dc_layer):
        """The internal trust map is [B, 1, H, W] and lives in (0, 1)."""
        measured_complex = torch.complex(torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8))
        lam = dc_layer._compute_lambda(measured_complex)
        assert lam.shape == (2, 1, 8, 8)
        assert (lam > 0.0).all()
        assert (lam < 1.0).all()

    def test_lambda_shared_across_coils(self, dc_layer):
        """Trust map averages magnitude over coils so it depends only on geometry."""
        # Two coil counts -> same lambda shape (always [B, 1, H, W]).
        m1 = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        m4 = torch.complex(torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8))
        assert dc_layer._compute_lambda(m1).shape == (1, 1, 8, 8)
        assert dc_layer._compute_lambda(m4).shape == (1, 1, 8, 8)

    def test_coords_cache_rebuilds_on_shape_change(self, dc_layer):
        """Coordinate buffers (radius, angle) are rebuilt only on (H, W) change."""
        m_8 = torch.complex(torch.randn(1, 1, 8, 8), torch.randn(1, 1, 8, 8))
        dc_layer._compute_lambda(m_8)
        cached_radius_8 = dc_layer._coord_radius.clone()
        # Same shape -> cache reused
        dc_layer._compute_lambda(m_8)
        assert torch.equal(dc_layer._coord_radius, cached_radius_8)
        # Different shape -> cache rebuilt
        m_16 = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))
        dc_layer._compute_lambda(m_16)
        assert dc_layer._coord_radius.shape == (16, 16)

    def test_mask_respected_unsampled_locations_unchanged(self, dc_layer):
        """At unsampled k-locations (mask=0) the prediction should be unchanged."""
        kspace_pred = torch.randn(1, 2, 8, 8)
        measured = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        # Sample only the central 2x2; mark unsampled locations clearly.
        mask[..., 3:5, 3:5] = 1.0
        out = dc_layer(
            kspace_pred,
            measured_kspace=measured,
            mask=mask,
            is_kspace_domain=True,
        )
        # Convert both to complex for direct comparison at unsampled cells.
        pred_c = torch.complex(kspace_pred[:, 0::2], kspace_pred[:, 1::2])
        out_c = torch.complex(out[:, 0::2], out[:, 1::2])
        unsampled = mask.bool().expand_as(pred_c).logical_not()
        assert torch.allclose(out_c[unsampled], pred_c[unsampled], atol=1e-4)

    def test_gradient_flows_kspace_path(self, dc_layer):
        """Backward through the k-space path yields finite gradients on both
        the prediction and the KAN parameters."""
        kspace_pred = torch.randn(1, 2, 8, 8, requires_grad=True)
        measured = torch.randn(1, 2, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 3:5, 3:5] = 1.0
        out = dc_layer(
            kspace_pred,
            measured_kspace=measured,
            mask=mask,
            is_kspace_domain=True,
        )
        out.sum().backward()
        assert kspace_pred.grad is not None
        assert torch.isfinite(kspace_pred.grad).all()
        # KAN params received gradient because lambda depends on log|y|.
        kan_grads = [p.grad for p in dc_layer.kan.parameters() if p.grad is not None]
        assert len(kan_grads) > 0
        assert all(torch.isfinite(g).all() for g in kan_grads)

    def test_complex_input_returns_complex(self, dc_layer):
        """Complex input tensor -> complex output (no real-stack repacking)."""
        kspace_complex = torch.complex(
            torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
        )
        measured_complex = torch.complex(
            torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8)
        )
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., 3:5, 3:5] = 1.0
        out = dc_layer(
            kspace_complex,
            measured_kspace=measured_complex,
            mask=mask,
            is_kspace_domain=True,
        )
        assert out.is_complex()
        assert out.shape == kspace_complex.shape

    def test_trust_map_telemetry_populated_after_forward(self, dc_layer):
        """The 4-element _last_trust_stats buffer is filled after a forward."""
        # Buffer starts at zeros
        assert torch.equal(dc_layer._last_trust_stats, torch.zeros(4))
        # Use a larger spatial size so the center mask (radius < 0.08) can
        # actually pick up some cells.
        measured = torch.complex(torch.randn(1, 2, 32, 32), torch.randn(1, 2, 32, 32))
        dc_layer._compute_lambda(measured)
        stats = dc_layer._last_trust_stats
        assert stats.shape == (4,)
        # All four stats should be in (0, 1) range from sigmoid; std bounded
        # by 0.5 (a Bernoulli-like distribution maxes out there).
        for i, name in enumerate(("center", "periphery", "mean")):
            assert 0.0 < float(stats[i]) < 1.0, f"{name} out of range: {stats[i]}"
        assert 0.0 <= float(stats[3]) < 0.6  # std

    def test_trust_map_zero_when_no_forward(self, dc_layer):
        """Buffer stays at zeros until _compute_lambda is called."""
        assert torch.equal(dc_layer._last_trust_stats, torch.zeros(4))
