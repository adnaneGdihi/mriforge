"""Tests for external StarGAN v2 arch port + checkpoint state extraction (Task 3).

Covers:
- ClovaaiStarGANv2Generator: forward pass shape and Tanh output range.
- ClovaaiMappingNetwork: forward pass shape.
- extract_generator_state: all supported checkpoint layouts + fail-loud on unknown.
- Round-trip: netG.-wrapped state dict loads into our cyclegan_generator with strict=True.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.evaluation.mrixfields_baselines.original_arch import (
    ClovaaiMappingNetwork,
    ClovaaiStarGANv2Generator,
    extract_generator_state,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_generator() -> ClovaaiStarGANv2Generator:
    """Small StarGAN v2 generator (img_size=64, max_conv_dim=128) for fast tests."""
    return ClovaaiStarGANv2Generator(
        img_size=64, style_dim=64, max_conv_dim=128, input_nc=1
    ).eval()


@pytest.fixture(scope="module")
def small_mapping_network() -> ClovaaiMappingNetwork:
    """Small mapping network (num_domains=15 as per spec)."""
    return ClovaaiMappingNetwork(latent_dim=16, style_dim=64, num_domains=15).eval()


# ---------------------------------------------------------------------------
# Generator + MappingNetwork forward-pass tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mapping_network_output_shape(small_mapping_network):
    """MappingNetwork(z, y) → (B, style_dim) = (2, 64)."""
    z = torch.randn(2, 16)
    y = torch.tensor([3, 7])
    with torch.no_grad():
        s = small_mapping_network(z, y)
    assert s.shape == (2, 64), f"Expected (2,64), got {s.shape}"


@pytest.mark.unit
def test_generator_output_shape(small_generator, small_mapping_network):
    """Generator(x, s) → (B, 1, H, W) = (2, 1, 64, 64)."""
    z = torch.randn(2, 16)
    y = torch.tensor([3, 7])
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        s = small_mapping_network(z, y)
        out = small_generator(x, s)
    assert out.shape == (2, 1, 64, 64), f"Expected (2,1,64,64), got {out.shape}"


@pytest.mark.unit
def test_generator_output_tanh_range(small_generator, small_mapping_network):
    """Generator output is in [-1, 1] (Tanh activation)."""
    z = torch.randn(2, 16)
    y = torch.tensor([0, 14])
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        s = small_mapping_network(z, y)
        out = small_generator(x, s)
    assert out.min().item() >= -1.0 - 1e-6, f"Output below -1: {out.min().item()}"
    assert out.max().item() <= 1.0 + 1e-6, f"Output above 1: {out.max().item()}"


# ---------------------------------------------------------------------------
# extract_generator_state tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_netg_prefix():
    """'model' key with netG. prefix: strip netG. and return generator weights."""
    w = torch.randn(3, 3)
    state_dict = {
        "model": {
            "netG.model.1.weight": w,
            "netD.discriminator.0.weight": torch.randn(3, 3),
        }
    }
    result = extract_generator_state(state_dict)
    assert set(result.keys()) == {"model.1.weight"}
    assert torch.allclose(result["model.1.weight"], w)


@pytest.mark.unit
def test_extract_netg_ab_prefix():
    """'model' key with netG_AB. prefix: strip netG_AB. and return generator weights."""
    w = torch.randn(4, 4)
    state_dict = {
        "model": {
            "netG_AB.model.1.weight": w,
            "netG_BA.model.1.weight": torch.randn(4, 4),
        }
    }
    result = extract_generator_state(state_dict)
    assert set(result.keys()) == {"model.1.weight"}
    assert torch.allclose(result["model.1.weight"], w)


@pytest.mark.unit
def test_extract_generator_key():
    """'generator' key: return its sub-dict directly."""
    w = torch.randn(2, 2)
    inner = {"layer.weight": w}
    state_dict = {"generator": inner, "discriminator": {}}
    result = extract_generator_state(state_dict)
    assert result is inner


@pytest.mark.unit
def test_extract_bare_dict_no_net_prefix():
    """Bare dict (no 'model'/'generator' key, no 'net*' top-level keys): returned as-is."""
    w = torch.randn(2, 2)
    state_dict = {"model.1.weight": w, "model.3.bias": torch.randn(2)}
    result = extract_generator_state(state_dict)
    assert result is state_dict


@pytest.mark.unit
def test_extract_unknown_layout_raises():
    """Dict with no 'model'/'generator' but with a 'net*' key raises ValueError."""
    state_dict = {
        "weird_key": torch.randn(2, 2),
        "netD_weight": torch.randn(2, 2),
    }
    with pytest.raises(ValueError, match="unrecognized checkpoint layout"):
        extract_generator_state(state_dict)


@pytest.mark.unit
def test_extract_model_no_netg_prefix_raises():
    """'model' key present but no netG./netG_AB. prefix: raises ValueError."""
    state_dict = {
        "model": {
            "netD.layer.weight": torch.randn(2, 2),
            "optimizer.state": torch.randn(2),
        }
    }
    with pytest.raises(ValueError, match="netG"):
        extract_generator_state(state_dict)


# ---------------------------------------------------------------------------
# Round-trip test: cyclegan_generator state dict loads via extract_generator_state
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cyclegan_generator_round_trip():
    """Wrap cyclegan_generator state dict with netG. prefix, extract, and reload strict=True."""
    import mriforge.models.generators.cycle_gan  # noqa: F401 — registers the model
    from mriforge.models.registry import get_model_class

    gen_cls = get_model_class("cyclegan_generator")

    # Build a generator and capture its state
    gen_a = gen_cls(input_nc=1, output_nc=1, ngf=64, n_blocks=9)
    sd = gen_a.state_dict()

    # Wrap as a checkpoint with netG. prefix (mimics CUT/CycleGAN checkpoint layout)
    wrapped = {"model": {f"netG.{k}": v for k, v in sd.items()}}

    # Extract
    extracted = extract_generator_state(wrapped)

    # Load into a fresh generator with strict=True (must not raise)
    gen_b = gen_cls(input_nc=1, output_nc=1, ngf=64, n_blocks=9)
    gen_b.load_state_dict(extracted, strict=True)

    # Verify a parameter was faithfully transferred
    first_key = next(iter(sd))
    assert torch.allclose(sd[first_key], gen_b.state_dict()[first_key]), (
        "Round-tripped parameter mismatch for key: " + first_key
    )
