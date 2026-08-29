"""Tests for the baseline generator loader (Task 4).

Covers:
- ResNet (cut/cyclegan) path: load checkpoint, forward runs, meta stamped.
- StarGAN path: load checkpoint, forward runs on correct crop size.
- StarGAN determinism: seed=0 twice → identical first-forward output; seed=1 differs.
- Fail-loud: target_domain=None → ValueError; unknown method → ValueError.

All tests run on CPU with tiny models so no GPU is required.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.infrastructure.evaluation.mrixfields_baselines.generator_loader import (
    LoadedBaseline,
    load_baseline_generator,
)
from mriforge.infrastructure.evaluation.mrixfields_baselines.original_arch import (
    ClovaaiMappingNetwork,
    ClovaaiStarGANv2Generator,
)
from mriforge.models.registry import get_model_class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RESNET_TINY = {"ngf": 8, "n_blocks": 1}
_STARGAN_TINY = {"img_size": 32, "max_conv_dim": 64}
_LATENT_DIM = 16
_NUM_DOMAINS = 15


def _make_resnet_ckpt(tmp_path):
    """Build a tiny ResNetGenerator and save it in the netG-wrapped format."""
    gen_cls = get_model_class("cyclegan_generator")
    gen = gen_cls(input_nc=1, output_nc=1, **_RESNET_TINY)
    ckpt_path = tmp_path / "resnet.pth"
    torch.save({"model": {f"netG.{k}": v for k, v in gen.state_dict().items()}}, ckpt_path)
    return ckpt_path, gen.state_dict()


def _make_stargan_ckpt(tmp_path):
    """Build tiny Clovaai nets and save as nets_ema checkpoint."""
    gen = ClovaaiStarGANv2Generator(
        style_dim=64, input_nc=1, **_STARGAN_TINY
    ).eval()
    mnet = ClovaaiMappingNetwork(
        latent_dim=_LATENT_DIM, style_dim=64, num_domains=_NUM_DOMAINS
    ).eval()
    ckpt_path = tmp_path / "stargan.pth"
    torch.save(
        {
            "nets_ema": {
                "generator": gen.state_dict(),
                "mapping_network": mnet.state_dict(),
            }
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# ResNet (cut / cyclegan) path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resnet_returns_loaded_baseline(tmp_path):
    """load_baseline_generator('cut') → LoadedBaseline with correct model_type."""
    ckpt, _ = _make_resnet_ckpt(tmp_path)
    result = load_baseline_generator("cut", ckpt, "cpu", resnet_kwargs=_RESNET_TINY)
    assert isinstance(result, LoadedBaseline)
    assert result.model_type == "resnet"
    assert result.crop_size is None


@pytest.mark.unit
def test_resnet_forward_runs(tmp_path):
    """ResNet forward pass on [1,1,16,16] completes without error."""
    ckpt, _ = _make_resnet_ckpt(tmp_path)
    result = load_baseline_generator("cut", ckpt, "cpu", resnet_kwargs=_RESNET_TINY)
    x = torch.randn(1, 1, 16, 16)
    out = result.forward(x)
    assert out.shape == (1, 1, 16, 16)


@pytest.mark.unit
def test_resnet_meta_stamped(tmp_path):
    """ResNet meta dict contains method and ngf."""
    ckpt, _ = _make_resnet_ckpt(tmp_path)
    result = load_baseline_generator("cut", ckpt, "cpu", resnet_kwargs=_RESNET_TINY)
    assert result.meta["method"] == "cut"
    assert "ngf" in result.meta
    assert result.meta["ngf"] == _RESNET_TINY["ngf"]
    assert "ckpt" in result.meta


@pytest.mark.unit
def test_cyclegan_method_also_works(tmp_path):
    """load_baseline_generator('cyclegan') uses the same ResNet path."""
    ckpt, _ = _make_resnet_ckpt(tmp_path)
    result = load_baseline_generator("cyclegan", ckpt, "cpu", resnet_kwargs=_RESNET_TINY)
    assert result.model_type == "resnet"


# ---------------------------------------------------------------------------
# StarGAN path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stargan_returns_loaded_baseline(tmp_path):
    """load_baseline_generator('stargan_v2') → LoadedBaseline with crop_size set."""
    ckpt = _make_stargan_ckpt(tmp_path)
    result = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    assert isinstance(result, LoadedBaseline)
    assert result.model_type == "stargan_v2"
    img_size = _STARGAN_TINY["img_size"]
    assert result.crop_size == (img_size, img_size)


@pytest.mark.unit
def test_stargan_forward_runs(tmp_path):
    """StarGAN forward pass on [1,1,32,32] completes without error."""
    ckpt = _make_stargan_ckpt(tmp_path)
    result = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    img_size = _STARGAN_TINY["img_size"]
    x = torch.randn(1, 1, img_size, img_size)
    out = result.forward(x)
    assert out.shape == (1, 1, img_size, img_size)


@pytest.mark.unit
def test_stargan_determinism_same_seed(tmp_path):
    """Two loads with seed=0 give identical first-forward output."""
    ckpt = _make_stargan_ckpt(tmp_path)
    img_size = _STARGAN_TINY["img_size"]
    x = torch.randn(1, 1, img_size, img_size)

    r1 = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    r2 = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    with torch.no_grad():
        out1 = r1.forward(x)
        out2 = r2.forward(x)
    assert torch.allclose(out1, out2), "seed=0 loads should produce identical output"


@pytest.mark.unit
def test_stargan_different_seeds_differ(tmp_path):
    """Loads with seed=0 and seed=1 give different first-forward output."""
    ckpt = _make_stargan_ckpt(tmp_path)
    img_size = _STARGAN_TINY["img_size"]
    x = torch.randn(1, 1, img_size, img_size)

    r0 = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    r1 = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=1, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    with torch.no_grad():
        out0 = r0.forward(x)
        out1 = r1.forward(x)
    assert not torch.allclose(out0, out1), "Different seeds should produce different output"


@pytest.mark.unit
def test_stargan_batch_forward(tmp_path):
    """StarGAN forward handles batch>1 via s.expand (style broadcast over the batch)."""
    ckpt = _make_stargan_ckpt(tmp_path)
    img_size = _STARGAN_TINY["img_size"]
    result = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=0, target_domain=4,
        stargan_kwargs=_STARGAN_TINY,
    )
    x = torch.randn(4, 1, img_size, img_size)
    with torch.no_grad():
        out = result.forward(x)
    assert out.shape == (4, 1, img_size, img_size)


@pytest.mark.unit
def test_style_isolated_from_global_rng(tmp_path):
    """Style sampling uses a LOCAL generator: with the same ``seed`` the bound style
    (hence the output) is identical regardless of the global torch RNG state.

    (The loader legitimately advances the global RNG while *constructing* the nets via
    weight init, but those weights are overwritten by ``load_state_dict``; only the style
    ``z`` must be reproducible, and it is, because it draws from a seeded local generator.)
    """
    ckpt = _make_stargan_ckpt(tmp_path)
    img_size = _STARGAN_TINY["img_size"]
    x = torch.randn(1, 1, img_size, img_size)
    torch.manual_seed(0)
    r_a = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=5, target_domain=4, stargan_kwargs=_STARGAN_TINY,
    )
    torch.manual_seed(999)
    r_b = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=5, target_domain=4, stargan_kwargs=_STARGAN_TINY,
    )
    with torch.no_grad():
        out_a, out_b = r_a.forward(x), r_b.forward(x)
    assert torch.allclose(out_a, out_b), "style must be reproducible from seed, not global RNG"


@pytest.mark.unit
def test_stargan_meta_stamped(tmp_path):
    """StarGAN meta contains target_domain, seed, and img_size."""
    ckpt = _make_stargan_ckpt(tmp_path)
    result = load_baseline_generator(
        "stargan_v2", ckpt, "cpu", seed=7, target_domain=3,
        stargan_kwargs=_STARGAN_TINY,
    )
    assert result.meta["target_domain"] == 3
    assert result.meta["seed"] == 7
    assert "img_size" in result.meta
    assert "ckpt" in result.meta


# ---------------------------------------------------------------------------
# Fail-loud cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stargan_target_domain_none_raises(tmp_path):
    """StarGAN without target_domain raises ValueError (fail-loud #9)."""
    ckpt = _make_stargan_ckpt(tmp_path)
    with pytest.raises(ValueError, match="target_domain"):
        load_baseline_generator(
            "stargan_v2", ckpt, "cpu", seed=0, target_domain=None,
            stargan_kwargs=_STARGAN_TINY,
        )


@pytest.mark.unit
def test_unknown_method_raises(tmp_path):
    """Unknown method raises ValueError (fail-loud #9)."""
    ckpt, _ = _make_resnet_ckpt(tmp_path)
    with pytest.raises(ValueError, match="unknown method"):
        load_baseline_generator("pix2pix", ckpt, "cpu")
