"""WS-5 ADAPT-002: ``MagnitudeFromComplex`` must not silently pass an
un-magnitudeable odd (>1) channel tensor through — it declares
``bridges_to={'domain': 'image'}``, so an input it can't reduce is a config bug.
"""

from __future__ import annotations

import pytest
import torch

from mriforge.data.adapters.channels import MagnitudeFromComplex


def test_even_channels_reduce_to_magnitude() -> None:
    out = MagnitudeFromComplex()(torch.randn(1, 4, 8, 8))  # 2 real/imag pairs
    assert out.shape == (1, 2, 8, 8)


def test_complex_input_reduces_to_real_magnitude() -> None:
    out = MagnitudeFromComplex()(torch.randn(1, 2, 8, 8, dtype=torch.complex64))
    assert not torch.is_complex(out)
    assert out.shape == (1, 2, 8, 8)


def test_single_channel_is_idempotent() -> None:
    x = torch.rand(1, 1, 8, 8)
    assert torch.equal(MagnitudeFromComplex()(x), x)


def test_odd_channels_gt_one_raises() -> None:
    with pytest.raises(ValueError, match="even channel count"):
        MagnitudeFromComplex()(torch.randn(1, 3, 8, 8))
