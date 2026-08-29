"""Unit tests for EnhancedDeepDiffusionUNet knob validation.

Pins the pitfall-#9 guards on ``time_embedding_type`` and ``noise_type``: an
unknown value must raise at construction instead of silently resolving to the
sinusoidal/gaussian defaults.
"""

import pytest
import torch
from torch import nn

from mriforge.models.diffusion.architectures.enhanced_deep_unet import (
    ChiSquareDiffusionUNet,
    EnhancedDeepDiffusionUNet,
    GaussianDiffusionUNet,
    GaussianKANDiffusionUNet,
    KANTimeEmbedding,
    create_diffusion_model,
)


def _tiny_model(**kwargs) -> EnhancedDeepDiffusionUNet:
    # model_channels must stay >= the ChannelAttention reduction (16), else
    # the attention squeeze conv is zero-element and forward cannot run.
    kwargs.setdefault("model_channels", 16)
    kwargs.setdefault("channel_mult", (1,))
    kwargs.setdefault("num_res_blocks", 1)
    return EnhancedDeepDiffusionUNet(in_channels=1, out_channels=1, **kwargs)


@pytest.mark.parametrize("noise_type", ["gaussian", "chi_square", "gaussian_kan"])
def test_valid_noise_types_construct(noise_type):
    model = _tiny_model(noise_type=noise_type)
    assert model.noise_type == noise_type


@pytest.mark.parametrize(
    "time_embedding_type,expected",
    [("sinusoidal", nn.Sequential), ("kan", KANTimeEmbedding)],
)
def test_valid_time_embedding_types_construct(time_embedding_type, expected):
    model = _tiny_model(time_embedding_type=time_embedding_type)
    assert isinstance(model.time_embed, expected)


def test_gaussian_forward_finite():
    model = _tiny_model(noise_type="gaussian")
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([5])
    out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_unknown_noise_type_raises():
    with pytest.raises(
        ValueError, match="'gaussian', 'chi_square', 'gaussian_kan'"
    ):
        _tiny_model(noise_type="rician")


def test_unknown_time_embedding_type_raises():
    with pytest.raises(ValueError, match="'sinusoidal', 'kan'"):
        _tiny_model(time_embedding_type="fourier")


@pytest.mark.parametrize(
    "subclass,noise_type",
    [
        (GaussianDiffusionUNet, "gaussian"),
        (ChiSquareDiffusionUNet, "chi_square"),
        (GaussianKANDiffusionUNet, "gaussian_kan"),
    ],
)
def test_variant_subclasses_still_construct(subclass, noise_type):
    model = subclass(
        in_channels=1,
        out_channels=1,
        model_channels=16,
        channel_mult=(1,),
        num_res_blocks=1,
    )
    assert model.noise_type == noise_type


def test_factory_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        create_diffusion_model(variant="poisson")
