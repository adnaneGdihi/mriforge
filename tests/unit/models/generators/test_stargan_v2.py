"""Unit tests for the StarGAN v2 model components (Task 8).

Covers the three from-scratch networks used by the MRIxFields any-to-any
FIELD-translation baseline (StarGAN v2, Choi et al. CVPR 2020), adapted to
single-channel MRI:

* ``StarGANv2Generator`` — AdaIN style-modulated encoder/decoder. The output
  MUST depend on the injected style ``s`` (an AdaIN that ignores ``s`` is a
  pitfall-#16 facade), so we assert a non-trivial delta between two styles.
* ``MappingNetwork`` — shared MLP + per-domain heads mapping ``z -> style``.
* ``StyleEncoder`` — shared conv trunk + per-domain heads mapping ``x -> style``.

Domain-index out-of-range must RAISE (no silent clamp — pitfall #9).
"""

import pytest
import torch

pytestmark = pytest.mark.unit

from spectramr.models.generators.stargan_v2 import (
    MappingNetwork,
    StarGANv2Generator,
    StyleEncoder,
)


def test_stargan_generator_shape_and_style_dependence() -> None:
    g = StarGANv2Generator(img_channels=1, style_dim=64)
    x = torch.rand(2, 1, 64, 64)
    s1 = torch.randn(2, 64)
    s2 = torch.randn(2, 64)
    y1, y2 = g(x, s1), g(x, s2)
    assert y1.shape == x.shape
    # Style genuinely modulates the output (anti-facade: AdaIN must read s).
    assert (y1 - y2).abs().mean() > 1e-4


def test_stargan_generator_non_square_and_odd_batch() -> None:
    # Resolution/aspect-agnostic: spatial shape is preserved for a 1-sample batch.
    g = StarGANv2Generator(img_channels=1, style_dim=32)
    x = torch.rand(1, 1, 64, 48)
    s = torch.randn(1, 32)
    out = g(x, s)
    assert out.shape == x.shape


def test_mapping_network_shape_and_domain_selection() -> None:
    m = MappingNetwork(latent_dim=16, style_dim=64, num_domains=5)
    z = torch.randn(3, 16)
    s = m(z, torch.tensor([0, 2, 4]))
    assert s.shape == (3, 64)
    # Different domain rows select different heads -> different styles.
    z2 = torch.randn(2, 16)
    s_low = m(z2, torch.tensor([0, 0]))
    s_high = m(z2, torch.tensor([4, 4]))
    assert (s_low - s_high).abs().mean() > 1e-6


def test_style_encoder_shape_and_domain_selection() -> None:
    e = StyleEncoder(img_channels=1, style_dim=64, num_domains=5)
    x = torch.rand(3, 1, 64, 64)
    s = e(x, torch.tensor([1, 3, 0]))
    assert s.shape == (3, 64)
    # Different domain rows select different heads.
    x2 = torch.rand(2, 1, 64, 64)
    s0 = e(x2, torch.tensor([0, 0]))
    s4 = e(x2, torch.tensor([4, 4]))
    assert (s0 - s4).abs().mean() > 1e-6


def test_out_of_range_domain_index_raises() -> None:
    # Pitfall #9: an unknown domain index must raise, never silently clamp.
    m = MappingNetwork(latent_dim=16, style_dim=8, num_domains=5)
    with pytest.raises(ValueError):
        m(torch.randn(1, 16), torch.tensor([5]))
    with pytest.raises(ValueError):
        m(torch.randn(1, 16), torch.tensor([-1]))

    e = StyleEncoder(img_channels=1, style_dim=8, num_domains=5)
    with pytest.raises(ValueError):
        e(torch.rand(1, 1, 32, 32), torch.tensor([7]))


def test_generator_registered_and_resolvable() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY, get_model_class, get_model_mode

    populate_model_registry()
    assert "stargan_v2_generator" in MODEL_REGISTRY
    assert get_model_class("stargan_v2_generator") is StarGANv2Generator
    assert get_model_mode("stargan_v2_generator") == "stargan_v2"
