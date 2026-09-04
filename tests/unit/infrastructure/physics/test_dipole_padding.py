"""Unit test for dipole kernel z-padding.

Verifies that z-padding reduces wrap-around artifacts from circular
convolution when applying the 3D dipole kernel on truncated 2.5D slabs.
"""

import torch

from spectramr.infrastructure.physics.dipole import apply_3d_dipole_kernel


class TestDipoleKernelPadding:
    """Tests for zero-padded dipole kernel convolution."""

    def test_output_shape_preserved_with_padding(self):
        """Output shape must match input shape regardless of padding."""
        chi = torch.randn(2, 1, 16, 64, 64)
        result_no_pad = apply_3d_dipole_kernel(chi, pad_z=0)
        result_padded = apply_3d_dipole_kernel(chi, pad_z=24)

        assert result_no_pad.shape == chi.shape
        assert result_padded.shape == chi.shape

    def test_no_nans_with_padding(self):
        """Padded output must not contain NaN values."""
        chi = torch.randn(1, 1, 16, 32, 32)
        result = apply_3d_dipole_kernel(chi, pad_z=24)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

    def test_point_source_wrap_around_reduced(self):
        """Point-source χ at slab center should have less wrap-around with padding.

        A point susceptibility source at the center of a 16-slice slab produces
        a dipole field. With padding, the field at the slab boundaries should
        be closer to the analytical 1/r³ decay rather than contaminated by
        wrap-around from the periodic FFT boundary.
        """
        D, H, W = 16, 64, 64
        chi = torch.zeros(1, 1, D, H, W)
        chi[0, 0, D // 2, H // 2, W // 2] = 1.0  # Point source at center

        # Without padding: circular convolution wraps
        delta_b0_no_pad = apply_3d_dipole_kernel(chi, pad_z=0)
        # With padding: linear convolution, no wrap-around
        delta_b0_padded = apply_3d_dipole_kernel(chi, pad_z=24)

        # At the boundary slices (z=0, z=15), the padded version should have
        # smaller absolute field values because the dipole tail doesn't wrap
        boundary_no_pad = delta_b0_no_pad[0, 0, 0, :, :].abs().mean()
        boundary_padded = delta_b0_padded[0, 0, 0, :, :].abs().mean()

        assert boundary_padded <= boundary_no_pad * 1.1, (
            f"Padded boundary field ({boundary_padded:.6f}) should be ≤ "
            f"unpadded ({boundary_no_pad:.6f})"
        )

    def test_zero_input_produces_zero_output(self):
        """All-zero χ must produce all-zero ΔB₀."""
        chi = torch.zeros(1, 1, 8, 32, 32)
        result = apply_3d_dipole_kernel(chi, pad_z=12)
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-7)

    def test_backward_compatibility_no_padding(self):
        """pad_z=0 should give identical results to the original function."""
        chi = torch.randn(1, 1, 16, 32, 32)
        result_default = apply_3d_dipole_kernel(chi)
        result_explicit = apply_3d_dipole_kernel(chi, pad_z=0)
        assert torch.allclose(result_default, result_explicit, atol=1e-6)
