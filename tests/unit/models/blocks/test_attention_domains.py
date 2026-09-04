"""Tests for the feature-domain SSOT (models/blocks/attention_domains.py)."""

import pytest
import torch

from spectramr.models.blocks.attention_domains import (
    ATTENTION_DOMAIN_SUPPORT,
    COMPLEX_UNET_BLOCK_ATTENTION,
    DOMAIN_AWARE_ATTENTION,
    FEATURE_DOMAINS,
    complex_to_interleaved,
    interleaved_to_complex,
    validate_feature_domain,
)

pytestmark = pytest.mark.unit


class TestValidateFeatureDomain:
    def test_accepts_both_valid_values(self):
        assert validate_feature_domain("kspace") == "kspace"
        assert validate_feature_domain("image") == "image"

    def test_normalises_case(self):
        assert validate_feature_domain("KSPACE") == "kspace"
        assert validate_feature_domain("Image") == "image"

    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="feature_domain"):
            validate_feature_domain("frequency")

    def test_raises_on_none_string(self):
        with pytest.raises(ValueError, match="feature_domain"):
            validate_feature_domain("none")


class TestSSOTMaps:
    def test_support_map_covers_every_attention_type(self):
        """Drift guard: the SSOT must track the AttentionType enum exactly."""
        from spectramr.models.reconstruction.unet import AttentionType

        enum_values = {e.value for e in AttentionType}
        assert set(ATTENTION_DOMAIN_SUPPORT) == enum_values

    def test_domain_aware_is_subset_of_support(self):
        assert DOMAIN_AWARE_ATTENTION.issubset(ATTENTION_DOMAIN_SUPPORT)

    def test_complex_unet_dispatch_is_subset_of_support(self):
        assert COMPLEX_UNET_BLOCK_ATTENTION.issubset(ATTENTION_DOMAIN_SUPPORT)

    def test_spatial_absent_from_complex_unet_dispatch(self):
        """'spatial' exists only in the up-block dispatch — ComplexUNet
        builds both blocks, so it must not be advertised for it."""
        assert "spatial" not in COMPLEX_UNET_BLOCK_ATTENTION
        assert "spatial" in ATTENTION_DOMAIN_SUPPORT

    def test_all_supported_domains_are_valid(self):
        for name, domains in ATTENTION_DOMAIN_SUPPORT.items():
            assert domains <= FEATURE_DOMAINS, name
            assert domains, f"{name} supports no domain"

    def test_dual_domain_family_supports_both(self):
        for name in DOMAIN_AWARE_ATTENTION:
            assert ATTENTION_DOMAIN_SUPPORT[name] == FEATURE_DOMAINS


class TestLayoutHelpers:
    def test_roundtrip(self):
        x = torch.randn(2, 8, 16, 16)
        assert torch.equal(complex_to_interleaved(interleaved_to_complex(x)), x)

    def test_complex_roundtrip(self):
        h = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))
        assert torch.equal(interleaved_to_complex(complex_to_interleaved(h)), h)

    def test_interleaving_order(self):
        """x[:, 0::2] is real, x[:, 1::2] is imag."""
        x = torch.zeros(1, 4, 2, 2)
        x[:, 0] = 1.0  # real of complex-ch 0
        x[:, 1] = 2.0  # imag of complex-ch 0
        x[:, 2] = 3.0  # real of complex-ch 1
        x[:, 3] = 4.0  # imag of complex-ch 1
        h = interleaved_to_complex(x)
        assert torch.equal(h[:, 0].real, torch.full((1, 2, 2), 1.0))
        assert torch.equal(h[:, 0].imag, torch.full((1, 2, 2), 2.0))
        assert torch.equal(h[:, 1].real, torch.full((1, 2, 2), 3.0))
        assert torch.equal(h[:, 1].imag, torch.full((1, 2, 2), 4.0))

    def test_odd_channels_raise(self):
        with pytest.raises(ValueError, match="even channels"):
            interleaved_to_complex(torch.randn(1, 3, 4, 4))
