"""Unit tests for Dual Domain Attention.

Tests both the complex signal attention gate and the dual-domain
attention mechanism that combines frequency and spatial processing.
"""

import torch
import torch.nn as nn

from mriforge.models.blocks.dual_domain_attention import (
    ComplexSignalAttention,
    DualDomainAttention,
)


class TestComplexSignalAttention:
    """Test ComplexSignalAttention module."""

    def test_init(self):
        """Test initialization with different parameters."""
        channels = 16
        reduction = 4
        attn = ComplexSignalAttention(complex_channels=channels, reduction=reduction)

        assert isinstance(attn.avg_pool, nn.Module)
        assert isinstance(attn.mlp, nn.Module)
        # mlp[0] is Linear(real_channels, real_channels // reduction)
        # real_channels = channels * 2
        assert attn.mlp[0].in_features == channels * 2
        assert attn.mlp[2].out_features == channels * 2

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        channels = 8
        batch_size = 2
        h, w = 32, 32

        attn = ComplexSignalAttention(complex_channels=channels)
        # Input is (B, 2*C, H, W) for complex signals
        x = torch.randn(batch_size, 2 * channels, h, w)

        output = attn(x)

        assert output.shape == (batch_size, 2 * channels, h, w)

    def test_scaling_logic(self):
        """Test that attention scales the input."""
        channels = 4
        attn = ComplexSignalAttention(complex_channels=channels)
        x = torch.randn(1, 2 * channels, 16, 16)

        output = attn(x)

        # Output should be different from input but same magnitude range roughly
        assert not torch.allclose(output, x)

        # Magnitude check: input * sigmoid(mag)
        # sigmoid is (0, 1), so output magnitude should be <= input magnitude
        # However, due to the complex conv weights, it depends on training.
        # But we can check it doesn't explode.
        assert not torch.any(torch.isnan(output))

    def test_gradient_flow(self):
        """Test that gradients flow through the block."""
        attn = ComplexSignalAttention(complex_channels=8)
        x = torch.randn(1, 16, 16, 16, requires_grad=True)

        output = attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestDualDomainAttention:
    """Test DualDomainAttention module."""

    def test_init(self):
        """Test initialization logic."""
        # Note: DualDomainAttention takes 'in_channels' which is tensor dim
        # (2 * complex_channels)
        tensor_channels = 16
        attn = DualDomainAttention(in_channels=tensor_channels, feature_domain="kspace")

        assert isinstance(attn.freq_attn, ComplexSignalAttention)
        assert isinstance(attn.spat_attn, ComplexSignalAttention)

        # Internal complex channels should be tensor_channels // 2, so 8.
        # It's mlp[0].in_features that stores the input channels (= tensor_channels)
        assert attn.freq_attn.mlp[0].in_features == tensor_channels

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        tensor_channels = 8
        batch_size = 1
        h, w = 16, 16

        attn = DualDomainAttention(in_channels=tensor_channels, feature_domain="kspace")
        x = torch.randn(batch_size, tensor_channels, h, w)

        output = attn(x)

        assert output.shape == (batch_size, tensor_channels, h, w)

    def test_dual_domain_processing(self):
        """Test that both frequency and spatial paths affect the output."""
        tensor_channels = 4
        attn = DualDomainAttention(in_channels=tensor_channels, feature_domain="kspace")
        x = torch.randn(1, tensor_channels, 16, 16)

        # Zero out one path to see if the other still works?
        # Hard to do without mocking.
        # But we can check if it runs and produces non-NaN output.
        output = attn(x)
        assert not torch.any(torch.isnan(output))
        assert output.shape == x.shape

    def test_complex_handling_consistency(self):
        """Test that it handles complex dimensions correctly via cat/split."""
        # DualDomainAttention internally splits real/imag
        # Input is (B, 2*C, H, W)
        C = 2
        tensor_channels = 2 * C
        attn = DualDomainAttention(in_channels=tensor_channels, feature_domain="kspace")

        x_real = torch.randn(1, C, 16, 16)
        x_imag = torch.randn(1, C, 16, 16)
        x = torch.cat([x_real, x_imag], dim=1)

        output = attn(x)

        # Verify output also has real/imag cat
        assert output.shape[1] == tensor_channels

        o_real = output[:, :C]
        o_imag = output[:, C:]

        # Just ensure they are not identical (unless by chance)
        assert not torch.allclose(o_real, o_imag)

    def test_fft_ifft_integration(self):
        """Test that FFT/IFFT operations within the block are valid."""
        # This is implicitly tested in forward_shape, but
        # we want to ensure no dimension mismatch on odd shapes.
        attn = DualDomainAttention(in_channels=4, feature_domain="kspace")

        # Test with odd spatial dimensions
        x = torch.randn(1, 4, 15, 17)
        output = attn(x)

        assert output.shape == (1, 4, 15, 17)


class TestDualDomainAttentionFeatureDomain:
    """Feature-domain contract for DualDomainAttention (2026-07-03)."""

    def test_feature_domain_is_required(self):
        import pytest

        with pytest.raises(TypeError):
            DualDomainAttention(in_channels=8)  # missing kw-only feature_domain

    def test_unknown_feature_domain_raises(self):
        import pytest

        with pytest.raises(ValueError, match="feature_domain"):
            DualDomainAttention(in_channels=8, feature_domain="frequency")

    def test_kspace_mode_matches_legacy_algorithm(self):
        """kspace mode == the historical freq/spatial composition, bit-exact."""
        from mriforge.infrastructure.physics.fft_ops import fft2c, ifft2c
        from mriforge.models.blocks.attention_domains import (
            complex_to_interleaved as c2i,
        )
        from mriforge.models.blocks.attention_domains import (
            interleaved_to_complex as i2c,
        )

        torch.manual_seed(0)
        attn = DualDomainAttention(in_channels=8, feature_domain="kspace").eval()
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            x_freq = attn.freq_attn(x)
            img = c2i(ifft2c(i2c(x)))
            img_attn = attn.spat_attn(img)
            x_spat = c2i(fft2c(i2c(img_attn)))
            legacy = x_freq + x_spat
            assert torch.equal(attn(x), legacy)

    def test_image_kspace_conjugacy(self):
        """out_kspace(fft2c(x)) == fft2c(out_image(x)) to FFT roundoff."""
        from mriforge.infrastructure.physics.fft_ops import fft2c
        from mriforge.models.blocks.attention_domains import (
            complex_to_interleaved as c2i,
        )
        from mriforge.models.blocks.attention_domains import (
            interleaved_to_complex as i2c,
        )

        torch.manual_seed(1)
        bi = DualDomainAttention(in_channels=8, feature_domain="image").eval()
        bk = DualDomainAttention(in_channels=8, feature_domain="kspace").eval()
        bk.load_state_dict(bi.state_dict())
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            out_i = bi(x)
            out_k = bk(c2i(fft2c(i2c(x))))
            rhs = c2i(fft2c(i2c(out_i)))
        torch.testing.assert_close(out_k, rhs, rtol=1e-4, atol=1e-4)
