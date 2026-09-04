"""Regression tests for ``configurable_architecture`` activation resolution.

Pitfall #9 (CLAUDE.md non-negotiable #3): an ``activation`` string outside the
advertised ``{relu, silu, gelu, leaky_relu}`` set must RAISE, not silently
degrade to ReLU. The pre-fix ``activations.get(activation, nn.ReLU(...))``
swallowed typos / YAML-sourced values and trained with a different
non-linearity than requested. These tests lock the fail-loud contract and
the no-numerics-change guarantee for the four valid strings.

CPU-only, tiny tensors, fixed seed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.latent.configurable_architecture import (
    DecoderConfig,
    EncoderConfig,
    FlexibleDecoder,
    FlexibleEncoder,
    LatentConfig,
    ResBlock,
)

_VALID_ACTIVATIONS = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
}


def test_flexible_encoder_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        FlexibleEncoder(EncoderConfig(activation="swish"), LatentConfig())


def test_flexible_decoder_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        FlexibleDecoder(DecoderConfig(activation="mish"), LatentConfig())


def test_resblock_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        ResBlock(8, 8, activation="swish")


@pytest.mark.parametrize("name,cls", list(_VALID_ACTIVATIONS.items()))
def test_resblock_valid_activation_resolves(name: str, cls: type[nn.Module]) -> None:
    # Positive case: each advertised activation resolves to the expected
    # nn.Module type (locks the no-numerics-change guarantee for valid configs).
    block = ResBlock(8, 8, activation=name)
    assert isinstance(block.activation, cls)


@pytest.mark.parametrize("name", list(_VALID_ACTIVATIONS))
def test_flexible_encoder_valid_activation_constructs(name: str) -> None:
    # All four advertised activations must build the encoder without error.
    enc = FlexibleEncoder(EncoderConfig(activation=name), LatentConfig())
    assert isinstance(enc, FlexibleEncoder)


def test_error_message_enumerates_valid_set() -> None:
    torch.manual_seed(0)
    with pytest.raises(ValueError) as excinfo:
        ResBlock(8, 8, activation="swish")
    msg = str(excinfo.value)
    for valid in _VALID_ACTIVATIONS:
        assert valid in msg
