"""Unit tests for SimpleEnhancedDiffusionUNet knob validation.

Pins the pitfall-#9 guards on ``time_embedding_type`` and ``noise_type``: an
unknown value must raise at construction instead of silently resolving to the
sinusoidal/gaussian defaults (a YAML advertising ``noise_type: rician`` would
otherwise run a plain gaussian model without any signal that the knob was
dropped).
"""

import pytest
import torch

from spectramr.models.diffusion.simple_enhanced_diffusion import (
    SimpleChiSquareDiffusionUNet,
    SimpleEnhancedDiffusionUNet,
    SimpleGaussianDiffusionUNet,
    SimpleGaussianKANDiffusionUNet,
    SimpleKANTimeEmbedding,
    SimpleTimeEmbedding,
)


def _tiny_model(**kwargs) -> SimpleEnhancedDiffusionUNet:
    kwargs.setdefault("model_channels", 8)
    kwargs.setdefault("channel_mult", (1,))
    kwargs.setdefault("num_res_blocks", 1)
    return SimpleEnhancedDiffusionUNet(in_channels=1, out_channels=1, **kwargs)


@pytest.mark.parametrize("noise_type", ["gaussian", "chi_square", "gaussian_kan"])
def test_valid_noise_types_construct_and_run(noise_type):
    model = _tiny_model(noise_type=noise_type)
    x = torch.randn(1, 1, 8, 8)
    t = torch.tensor([5])
    out = model(x, t)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "time_embedding_type,expected",
    [("sinusoidal", SimpleTimeEmbedding), ("kan", SimpleKANTimeEmbedding)],
)
def test_valid_time_embedding_types_construct(time_embedding_type, expected):
    model = _tiny_model(time_embedding_type=time_embedding_type)
    assert isinstance(model.time_embed, expected)


def test_unknown_noise_type_raises():
    with pytest.raises(
        ValueError, match="'gaussian', 'chi_square', 'gaussian_kan'"
    ):
        _tiny_model(noise_type="rician")


def test_unknown_time_embedding_type_raises():
    with pytest.raises(ValueError, match="'sinusoidal', 'kan'"):
        _tiny_model(time_embedding_type="fourier")


@pytest.mark.parametrize(
    "subclass",
    [
        SimpleGaussianDiffusionUNet,
        SimpleChiSquareDiffusionUNet,
        SimpleGaussianKANDiffusionUNet,
    ],
)
def test_registered_variants_still_construct(subclass):
    model = subclass(
        in_channels=1,
        out_channels=1,
        model_channels=8,
        channel_mult=(1,),
        num_res_blocks=1,
    )
    assert model.noise_type in ("gaussian", "chi_square", "gaussian_kan")
