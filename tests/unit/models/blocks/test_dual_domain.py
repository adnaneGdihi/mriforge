"""Unit tests for DualDomainBlock (feature-domain aware).

DualDomainBlock runs unconditionally as the ``unet_block`` in every ComplexUNet
down/up block, yet had no dedicated test before the 2026-07-03 feature-domain
change. These tests pin its shape contract in both conv variants and both
feature domains, the legacy-equivalence of kspace mode, and the image<->kspace
conjugacy identity.
"""

import pytest
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.models.blocks.attention_domains import (
    complex_to_interleaved as c2i,
)
from spectramr.models.blocks.attention_domains import (
    interleaved_to_complex as i2c,
)
from spectramr.models.blocks.dual_domain import DualDomainBlock

pytestmark = pytest.mark.unit


class TestConstruction:
    def test_feature_domain_is_required(self):
        with pytest.raises(TypeError):
            DualDomainBlock(8)  # missing kw-only feature_domain

    def test_unknown_feature_domain_raises(self):
        with pytest.raises(ValueError, match="feature_domain"):
            DualDomainBlock(8, feature_domain="frequency")


@pytest.mark.parametrize("feature_domain", ["kspace", "image"])
@pytest.mark.parametrize("use_complex_conv", [True, False])
class TestForward:
    def test_shape_preserved(self, feature_domain, use_complex_conv):
        blk = DualDomainBlock(
            8, use_complex_conv=use_complex_conv, feature_domain=feature_domain
        ).eval()
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            out = blk(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_time_embedding_path(self, feature_domain, use_complex_conv):
        blk = DualDomainBlock(
            8,
            use_complex_conv=use_complex_conv,
            time_embedding_dim=32,
            feature_domain=feature_domain,
        ).eval()
        x = torch.randn(2, 8, 16, 16)
        t = torch.randn(2, 32)
        with torch.no_grad():
            out = blk(x, t)
        assert out.shape == x.shape


class TestFeatureDomainSemantics:
    def test_kspace_mode_matches_legacy_algorithm(self):
        """kspace mode reproduces the historical k-path / image-path fusion."""
        torch.manual_seed(0)
        blk = DualDomainBlock(8, use_complex_conv=True, feature_domain="kspace").eval()
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            k_feat = blk.k_conv_2(blk.activation(blk.k_gn(blk.k_conv_1(x))))
            img = c2i(ifft2c(i2c(x)))
            i = blk.activation(blk.i_gn(blk.i_conv_1(img)))
            i_feat_spatial = blk.i_conv_2(i)
            i_feat = c2i(fft2c(i2c(i_feat_spatial)))
            legacy = blk.fusion_proj(torch.cat([k_feat, i_feat], dim=1))
            assert torch.equal(blk(x), legacy)

    def test_image_kspace_conjugacy(self):
        torch.manual_seed(2)
        bi = DualDomainBlock(8, use_complex_conv=True, feature_domain="image").eval()
        bk = DualDomainBlock(8, use_complex_conv=True, feature_domain="kspace").eval()
        bk.load_state_dict(bi.state_dict())
        x = torch.randn(2, 8, 16, 16)
        with torch.no_grad():
            out_i = bi(x)
            out_k = bk(c2i(fft2c(i2c(x))))
            rhs = c2i(fft2c(i2c(out_i)))
        torch.testing.assert_close(out_k, rhs, rtol=1e-4, atol=1e-4)
